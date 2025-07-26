/**
 * Concurrent RAG System Frontend Application
 */

class RAGApp {
    constructor() {
        this.apiBase = '/api';
        this.wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/updates`;
        this.token = localStorage.getItem('auth_token');
        this.user = null;
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        
        // Check if user is already logged in
        if (this.token) {
            try {
                await this.validateSession();
                this.showMainApp();
                this.connectWebSocket();
            } catch (error) {
                console.error('Session validation failed:', error);
                this.logout();
            }
        } else {
            this.showLogin();
        }
    }

    setupEventListeners() {
        // Login form
        const loginForm = document.getElementById('loginFormElement');
        if (loginForm) {
            loginForm.addEventListener('submit', this.handleLogin.bind(this));
        }

        // Logout button
        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', this.logout.bind(this));
        }

        // Navigation tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.closest('.nav-tab').dataset.tab;
                this.switchTab(tabName);
            });
        });

        // File upload
        this.setupFileUpload();

        // Query form
        const submitQueryBtn = document.getElementById('submitQueryBtn');
        if (submitQueryBtn) {
            submitQueryBtn.addEventListener('click', this.handleQuery.bind(this));
        }

        // Document filters
        const refreshDocuments = document.getElementById('refreshDocuments');
        if (refreshDocuments) {
            refreshDocuments.addEventListener('click', this.loadDocuments.bind(this));
        }

        const groupFilter = document.getElementById('groupFilter');
        const statusFilter = document.getElementById('statusFilter');
        if (groupFilter) groupFilter.addEventListener('change', this.loadDocuments.bind(this));
        if (statusFilter) statusFilter.addEventListener('change', this.loadDocuments.bind(this));
    }

    setupFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');

        if (!uploadArea || !fileInput || !uploadBtn) return;

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files).filter(file => 
                file.type === 'application/pdf'
            );
            this.handleFileSelection(files);
        });

        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            this.handleFileSelection(files);
        });

        uploadBtn.addEventListener('click', this.handleFileUpload.bind(this));
    }

    handleFileSelection(files) {
        if (files.length === 0) return;

        const fileList = document.getElementById('fileList');
        const selectedFiles = document.getElementById('selectedFiles');
        
        if (!fileList || !selectedFiles) return;

        // Clear previous selection
        selectedFiles.innerHTML = '';
        
        // Store files for upload
        this.selectedFiles = files;

        // Display selected files
        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="file-info">
                    <i class="fas fa-file-pdf file-icon"></i>
                    <div class="file-details">
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${this.formatFileSize(file.size)}</div>
                    </div>
                </div>
                <button class="remove-file" data-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;
            
            const removeBtn = fileItem.querySelector('.remove-file');
            removeBtn.addEventListener('click', () => {
                this.removeFile(index);
            });
            
            selectedFiles.appendChild(fileItem);
        });

        fileList.style.display = 'block';
    }

    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        
        if (this.selectedFiles.length === 0) {
            document.getElementById('fileList').style.display = 'none';
        } else {
            this.handleFileSelection(this.selectedFiles);
        }
    }

    async handleFileUpload() {
        const groupSelect = document.getElementById('groupSelect');
        const groupId = groupSelect.value;

        if (!groupId) {
            this.showToast('error', 'Error', 'Please select a group');
            return;
        }

        if (!this.selectedFiles || this.selectedFiles.length === 0) {
            this.showToast('error', 'Error', 'Please select files to upload');
            return;
        }

        const uploadProgress = document.getElementById('uploadProgress');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const uploadBtn = document.getElementById('uploadBtn');

        uploadProgress.style.display = 'block';
        uploadBtn.disabled = true;

        try {
            const formData = new FormData();
            this.selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            formData.append('group_id', groupId);

            const response = await fetch(`${this.apiBase}/documents/upload`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.token}`
                },
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error?.message || 'Upload failed');
            }

            const result = await response.json();
            
            // Simulate progress (real progress would come from WebSocket)
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 10;
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `${progress}%`;
                
                if (progress >= 100) {
                    clearInterval(progressInterval);
                    this.showToast('success', 'Upload Complete', 
                        `Successfully uploaded ${result.files_count} files`);
                    
                    // Reset form
                    this.resetUploadForm();
                    
                    // Refresh documents and jobs
                    this.loadDocuments();
                    this.loadJobs();
                }
            }, 200);

        } catch (error) {
            console.error('Upload error:', error);
            this.showToast('error', 'Upload Failed', error.message);
            uploadProgress.style.display = 'none';
            uploadBtn.disabled = false;
        }
    }

    resetUploadForm() {
        document.getElementById('fileList').style.display = 'none';
        document.getElementById('uploadProgress').style.display = 'none';
        document.getElementById('fileInput').value = '';
        document.getElementById('uploadBtn').disabled = false;
        this.selectedFiles = [];
    }

    async handleQuery() {
        const queryInput = document.getElementById('queryInput');
        const queryText = queryInput.value.trim();

        if (!queryText) {
            this.showToast('error', 'Error', 'Please enter a query');
            return;
        }

        const submitBtn = document.getElementById('submitQueryBtn');
        const queryResults = document.getElementById('queryResults');
        const queryResponse = document.getElementById('queryResponse');
        const queryStatus = document.getElementById('queryStatus');

        submitBtn.disabled = true;
        queryResults.style.display = 'block';
        queryResponse.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i>Processing your query...</div>';
        queryStatus.innerHTML = '<i class="fas fa-clock"></i> Status: Processing';

        try {
            const response = await fetch(`${this.apiBase}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.token}`
                },
                body: JSON.stringify({ query_text: queryText })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error?.message || 'Query failed');
            }

            const result = await response.json();
            
            // Poll for results (real updates would come from WebSocket)
            this.pollQueryResult(result.query_id);

        } catch (error) {
            console.error('Query error:', error);
            queryResponse.innerHTML = `<div class="error-message">Query failed: ${error.message}</div>`;
            queryStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Status: Failed';
            submitBtn.disabled = false;
        }
    }

    async pollQueryResult(queryId) {
        const maxAttempts = 30; // 30 seconds timeout
        let attempts = 0;

        const poll = async () => {
            try {
                const response = await fetch(`${this.apiBase}/query/${queryId}`, {
                    headers: {
                        'Authorization': `Bearer ${this.token}`
                    }
                });

                if (!response.ok) {
                    throw new Error('Failed to get query status');
                }

                const result = await response.json();
                const queryStatus = document.getElementById('queryStatus');
                const queryResponse = document.getElementById('queryResponse');
                const submitBtn = document.getElementById('submitQueryBtn');

                if (result.status === 'completed') {
                    queryResponse.innerHTML = result.result || 'No results found';
                    queryStatus.innerHTML = '<i class="fas fa-check-circle"></i> Status: Completed';
                    submitBtn.disabled = false;
                    return;
                } else if (result.status === 'failed') {
                    queryResponse.innerHTML = `<div class="error-message">Query failed: ${result.error || 'Unknown error'}</div>`;
                    queryStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Status: Failed';
                    submitBtn.disabled = false;
                    return;
                } else if (attempts < maxAttempts) {
                    attempts++;
                    setTimeout(poll, 1000);
                } else {
                    queryResponse.innerHTML = '<div class="error-message">Query timeout</div>';
                    queryStatus.innerHTML = '<i class="fas fa-clock"></i> Status: Timeout';
                    submitBtn.disabled = false;
                }
            } catch (error) {
                console.error('Polling error:', error);
                const queryResponse = document.getElementById('queryResponse');
                const queryStatus = document.getElementById('queryStatus');
                const submitBtn = document.getElementById('submitQueryBtn');
                
                queryResponse.innerHTML = `<div class="error-message">Error checking query status: ${error.message}</div>`;
                queryStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Status: Error';
                submitBtn.disabled = false;
            }
        };

        poll();
    }

    async handleLogin(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const loginError = document.getElementById('loginError');

        try {
            const response = await fetch(`${this.apiBase}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error?.message || 'Login failed');
            }

            const result = await response.json();
            this.token = result.access_token;
            this.user = {
                user_id: result.user_id,
                groups: result.groups
            };

            localStorage.setItem('auth_token', this.token);
            
            this.showMainApp();
            this.connectWebSocket();
            
        } catch (error) {
            console.error('Login error:', error);
            loginError.textContent = error.message;
            loginError.style.display = 'block';
        }
    }

    async logout() {
        try {
            if (this.token) {
                await fetch(`${this.apiBase}/auth/logout`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.token}`
                    }
                });
            }
        } catch (error) {
            console.error('Logout error:', error);
        }

        this.token = null;
        this.user = null;
        localStorage.removeItem('auth_token');
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.showLogin();
    }

    async validateSession() {
        const response = await fetch(`${this.apiBase}/auth/session`, {
            headers: {
                'Authorization': `Bearer ${this.token}`
            }
        });

        if (!response.ok) {
            throw new Error('Session validation failed');
        }

        const session = await response.json();
        this.user = {
            user_id: session.user_id,
            groups: session.groups
        };
    }

    showLogin() {
        document.getElementById('loginForm').style.display = 'flex';
        document.getElementById('mainApp').style.display = 'none';
        document.getElementById('userInfo').style.display = 'none';
    }

    showMainApp() {
        document.getElementById('loginForm').style.display = 'none';
        document.getElementById('mainApp').style.display = 'block';
        document.getElementById('userInfo').style.display = 'flex';
        document.getElementById('userName').textContent = this.user.user_id;
        
        this.populateGroupSelects();
        this.loadDocuments();
        this.loadJobs();
    }

    populateGroupSelects() {
        const groupSelect = document.getElementById('groupSelect');
        const groupFilter = document.getElementById('groupFilter');
        
        if (groupSelect) {
            groupSelect.innerHTML = '<option value="">Select a group...</option>';
            this.user.groups.forEach(group => {
                const option = document.createElement('option');
                option.value = group;
                option.textContent = group;
                groupSelect.appendChild(option);
            });
        }
        
        if (groupFilter) {
            groupFilter.innerHTML = '<option value="">All Groups</option>';
            this.user.groups.forEach(group => {
                const option = document.createElement('option');
                option.value = group;
                option.textContent = group;
                groupFilter.appendChild(option);
            });
        }
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        document.getElementById(`${tabName}Tab`).classList.add('active');

        // Load data for specific tabs
        if (tabName === 'documents') {
            this.loadDocuments();
        } else if (tabName === 'jobs') {
            this.loadJobs();
        }
    }

    async loadDocuments() {
        const documentsGrid = document.getElementById('documentsGrid');
        const documentsLoading = document.getElementById('documentsLoading');
        const groupFilter = document.getElementById('groupFilter');
        const statusFilter = document.getElementById('statusFilter');

        documentsLoading.style.display = 'flex';
        documentsGrid.innerHTML = '';

        try {
            const params = new URLSearchParams();
            if (groupFilter && groupFilter.value) params.append('group_id', groupFilter.value);
            if (statusFilter && statusFilter.value) params.append('status', statusFilter.value);
            params.append('limit', '50');

            const response = await fetch(`${this.apiBase}/documents?${params}`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to load documents');
            }

            const result = await response.json();
            this.displayDocuments(result.documents);

        } catch (error) {
            console.error('Error loading documents:', error);
            documentsGrid.innerHTML = `<div class="error-message">Failed to load documents: ${error.message}</div>`;
        } finally {
            documentsLoading.style.display = 'none';
        }
    }

    displayDocuments(documents) {
        const documentsGrid = document.getElementById('documentsGrid');
        
        if (documents.length === 0) {
            documentsGrid.innerHTML = '<div class="loading">No documents found</div>';
            return;
        }

        documentsGrid.innerHTML = documents.map(doc => `
            <div class="document-card">
                <div class="document-header">
                    <i class="fas fa-file-pdf document-icon"></i>
                    <div class="document-title">${doc.filename}</div>
                </div>
                <div class="document-meta">
                    <div class="document-meta-item">
                        <span>Group:</span>
                        <span>${doc.group_id}</span>
                    </div>
                    <div class="document-meta-item">
                        <span>Size:</span>
                        <span>${this.formatFileSize(doc.file_size)}</span>
                    </div>
                    <div class="document-meta-item">
                        <span>Uploaded:</span>
                        <span>${this.formatDate(doc.upload_date)}</span>
                    </div>
                    <div class="document-meta-item">
                        <span>Status:</span>
                        <span class="document-status status-${doc.processing_status}">
                            ${this.getStatusIcon(doc.processing_status)}
                            ${doc.processing_status}
                        </span>
                    </div>
                </div>
                <div class="document-actions">
                    <button class="btn btn-danger btn-small" onclick="app.deleteDocument('${doc.document_id}')">
                        <i class="fas fa-trash"></i>
                        Delete
                    </button>
                </div>
            </div>
        `).join('');
    }

    async deleteDocument(documentId) {
        if (!confirm('Are you sure you want to delete this document?')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/documents/${documentId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error?.message || 'Delete failed');
            }

            this.showToast('success', 'Document Deleted', 'Document deleted successfully');
            this.loadDocuments();

        } catch (error) {
            console.error('Delete error:', error);
            this.showToast('error', 'Delete Failed', error.message);
        }
    }

    async loadJobs() {
        const jobsList = document.getElementById('jobsList');
        const jobsLoading = document.getElementById('jobsLoading');
        const activeJobs = document.getElementById('activeJobs');
        const completedJobs = document.getElementById('completedJobs');
        const failedJobs = document.getElementById('failedJobs');

        jobsLoading.style.display = 'flex';
        jobsList.innerHTML = '';

        try {
            const response = await fetch(`${this.apiBase}/jobs`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to load jobs');
            }

            const result = await response.json();
            this.displayJobs(result.jobs);
            
            // Update summary
            const stats = this.calculateJobStats(result.jobs);
            activeJobs.textContent = stats.active;
            completedJobs.textContent = stats.completed;
            failedJobs.textContent = stats.failed;

        } catch (error) {
            console.error('Error loading jobs:', error);
            jobsList.innerHTML = `<div class="error-message">Failed to load jobs: ${error.message}</div>`;
        } finally {
            jobsLoading.style.display = 'none';
        }
    }

    calculateJobStats(jobs) {
        return jobs.reduce((stats, job) => {
            if (job.status === 'processing' || job.status === 'pending') {
                stats.active++;
            } else if (job.status === 'completed') {
                stats.completed++;
            } else if (job.status === 'failed') {
                stats.failed++;
            }
            return stats;
        }, { active: 0, completed: 0, failed: 0 });
    }

    displayJobs(jobs) {
        const jobsList = document.getElementById('jobsList');
        
        if (jobs.length === 0) {
            jobsList.innerHTML = '<div class="loading">No jobs found</div>';
            return;
        }

        jobsList.innerHTML = jobs.map(job => `
            <div class="job-card" data-job-id="${job.job_id}">
                <div class="job-header">
                    <div class="job-title">
                        ${this.getJobIcon(job.job_type)}
                        ${job.job_type} Job
                    </div>
                    <div class="job-time">${this.formatDate(job.created_at)}</div>
                </div>
                <div class="job-progress">
                    <div class="job-progress-bar">
                        <div class="job-progress-fill" style="width: ${job.progress * 100}%"></div>
                    </div>
                    <div class="job-progress-text">${Math.round(job.progress * 100)}% complete</div>
                </div>
                <div class="job-details">
                    <div class="job-info">
                        Status: <span class="document-status status-${job.status}">
                            ${this.getStatusIcon(job.status)}
                            ${job.status}
                        </span>
                    </div>
                    <div class="job-actions">
                        ${job.status === 'processing' || job.status === 'pending' ? 
                            `<button class="btn btn-danger btn-small" onclick="app.cancelJob('${job.job_id}')">
                                <i class="fas fa-stop"></i>
                                Cancel
                            </button>` : ''
                        }
                    </div>
                </div>
            </div>
        `).join('');
    }

    async cancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/jobs/${jobId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error?.message || 'Cancel failed');
            }

            this.showToast('success', 'Job Cancelled', 'Job cancelled successfully');
            this.loadJobs();

        } catch (error) {
            console.error('Cancel error:', error);
            this.showToast('error', 'Cancel Failed', error.message);
        }
    }

    connectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
        }

        try {
            // Add token as query parameter
            const wsUrlWithToken = `${this.wsUrl}?token=${encodeURIComponent(this.token)}`;
            this.websocket = new WebSocket(wsUrlWithToken);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
                
                // Start heartbeat
                this.startHeartbeat();
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('WebSocket message parse error:', error);
                }
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('disconnected');
                this.scheduleReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('disconnected');
            };

        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateConnectionStatus('disconnected');
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts && this.token) {
            this.reconnectAttempts++;
            this.updateConnectionStatus('connecting');
            
            setTimeout(() => {
                console.log(`Reconnecting WebSocket (attempt ${this.reconnectAttempts})`);
                this.connectWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }

    handleWebSocketMessage(message) {
        console.log('WebSocket message:', message);
        
        switch (message.type) {
            case 'job_update':
                this.handleJobUpdate(message.data);
                break;
            case 'notification':
                this.showToast(message.data.level, message.data.title, message.data.message);
                break;
            case 'error':
                console.error('WebSocket error:', message.error);
                this.showToast('error', 'Connection Error', message.error.message);
                break;
            default:
                console.log('Unknown WebSocket message type:', message.type);
        }
    }

    handleJobUpdate(jobData) {
        // Update job progress in real-time
        const jobCard = document.querySelector(`[data-job-id="${jobData.job_id}"]`);
        if (jobCard) {
            const progressFill = jobCard.querySelector('.job-progress-fill');
            const progressText = jobCard.querySelector('.job-progress-text');
            const statusElement = jobCard.querySelector('.document-status');
            
            if (progressFill) {
                progressFill.style.width = `${jobData.progress * 100}%`;
            }
            if (progressText) {
                progressText.textContent = `${Math.round(jobData.progress * 100)}% complete`;
            }
            if (statusElement) {
                statusElement.className = `document-status status-${jobData.status}`;
                statusElement.innerHTML = `${this.getStatusIcon(jobData.status)} ${jobData.status}`;
            }
        }
        
        // Refresh relevant tabs
        const activeTab = document.querySelector('.nav-tab.active').dataset.tab;
        if (activeTab === 'jobs') {
            this.loadJobs();
        } else if (activeTab === 'documents' && jobData.job_type === 'embedding') {
            this.loadDocuments();
        }
        
        // Show notification for completed jobs
        if (jobData.status === 'completed') {
            this.showToast('success', 'Job Completed', 
                `${jobData.job_type} job completed successfully`);
        } else if (jobData.status === 'failed') {
            this.showToast('error', 'Job Failed', 
                `${jobData.job_type} job failed: ${jobData.error || 'Unknown error'}`);
        }
    }

    startHeartbeat() {
        // Clear existing heartbeat
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
        
        // Send ping every 30 seconds
        this.heartbeatInterval = setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now()
                }));
            }
        }, 30000);
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }L = `${this.getStatusIcon(jobData.status)} ${jobData.status}`;
            }
        }
        
        // Refresh relevant tabs
        const activeTab = document.querySelector('.nav-tab.active').dataset.tab;
        if (activeTab === 'jobs') {
            this.loadJobs();
        } else if (activeTab === 'documents' && jobData.job_type === 'embedding') {
            this.loadDocuments();
        }
        
        // Show notification for completed jobs
        if (jobData.status === 'completed') {
            this.showToast('success', 'Job Completed', 
                `${jobData.job_type} job completed successfully`);
        } else if (jobData.status === 'failed') {
            this.showToast('error', 'Job Failed', 
                `${jobData.job_type} job failed: ${jobData.error || 'Unknown error'}`);
        }
    }

    startHeartbeat() {
        // Clear existing heartbeat
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
        
        // Send ping every 30 seconds
        this.heartbeatInterval = setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now()
                }));
            }
        }, 30000);
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }L = `${this.getStatusIcon(jobData.status)} ${jobData.status}`;
            }
        }
        
        // Refresh relevant tabs
        const activeTab = document.querySelector('.nav-tab.active').dataset.tab;
        if (activeTab === 'jobs') {
            this.loadJobs();
        } else if (activeTab === 'documents' && jobData.job_type === 'embedding') {
            this.loadDocuments();
        }
        
        // Show notification for completed jobs
        if (jobData.status === 'completed') {
            this.showToast('success', 'Job Completed', 
                `${jobData.job_type} job completed successfully`);
        } else if (jobData.status === 'failed') {
            this.showToast('error', 'Job Failed', 
                `${jobData.job_type} job failed: ${jobData.error || 'Unknown error'}`);
        }
    }


    updateConnectionStatus(status) {
        const connectionStatus = document.getElementById('connectionStatus');
        const connectionText = document.getElementById('connectionText');
        
        connectionStatus.className = `connection-status ${status}`;
        
        switch (status) {
            case 'connected':
                connectionText.textContent = 'Connected';
                break;
            case 'connecting':
                connectionText.textContent = 'Connecting...';
                break;
            case 'disconnected':
                connectionText.textContent = 'Disconnected';
                break;
        }
    }

    showToast(type, title, message) {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            toast.remove();
        });
        
        toastContainer.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }

    // Utility functions
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    getStatusIcon(status) {
        switch (status) {
            case 'processing':
            case 'pending':
                return '<i class="fas fa-spinner fa-spin"></i>';
            case 'completed':
                return '<i class="fas fa-check-circle"></i>';
            case 'failed':
                return '<i class="fas fa-exclamation-circle"></i>';
            default:
                return '<i class="fas fa-question-circle"></i>';
        }
    }

    getJobIcon(jobType) {
        switch (jobType) {
            case 'embedding':
                return '<i class="fas fa-file-upload"></i>';
            case 'query':
                return '<i class="fas fa-search"></i>';
            default:
                return '<i class="fas fa-cog"></i>';
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new RAGApp();
});L
 = `${this.getStatusIcon(jobData.status)} ${jobData.status}`;
            }
        }
        
        // Refresh relevant tabs
        const activeTab = document.querySelector('.nav-tab.active').dataset.tab;
        if (activeTab === 'jobs') {
            this.loadJobs();
        } else if (activeTab === 'documents' && jobData.job_type === 'embedding') {
            this.loadDocuments();
        }
        
        // Show notification for completed jobs
        if (jobData.status === 'completed') {
            this.showToast('success', 'Job Completed', 
                `${jobData.job_type} job completed successfully`);
        } else if (jobData.status === 'failed') {
            this.showToast('error', 'Job Failed', 
                `${jobData.job_type} job failed: ${jobData.error || 'Unknown error'}`);
        }
    }

    startHeartbeat() {
        // Clear existing heartbeat
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
        
        // Send ping every 30 seconds
        this.heartbeatInterval = setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now()
                }));
            }
        }, 30000);
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    updateConnectionStatus(status) {
        const connectionStatus = document.getElementById('connectionStatus');
        const connectionText = document.getElementById('connectionText');
        
        connectionStatus.className = `connection-status ${status}`;
        
        switch (status) {
            case 'connected':
                connectionText.textContent = 'Connected';
                break;
            case 'connecting':
                connectionText.textContent = 'Connecting...';
                break;
            case 'disconnected':
                connectionText.textContent = 'Disconnected';
                this.stopHeartbeat();
                break;
        }
    }

    showToast(type, title, message) {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        toast.innerHTML = `
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            toast.remove();
        });
        
        toastContainer.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
    }

    // Utility functions
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    getStatusIcon(status) {
        switch (status) {
            case 'processing':
            case 'pending':
                return '<i class="fas fa-spinner fa-spin"></i>';
            case 'completed':
                return '<i class="fas fa-check-circle"></i>';
            case 'failed':
                return '<i class="fas fa-exclamation-circle"></i>';
            default:
                return '<i class="fas fa-question-circle"></i>';
        }
    }

    getJobIcon(jobType) {
        switch (jobType) {
            case 'embedding':
                return '<i class="fas fa-file-upload"></i>';
            case 'query':
                return '<i class="fas fa-search"></i>';
            default:
                return '<i class="fas fa-cog"></i>';
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new RAGApp();
});