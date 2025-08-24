/**
 * Concurrent RAG System Frontend Application with Keycloak Integration
 */

class RAGApp {
    constructor() {
        this.apiBase = '/api';
        this.keycloak = null;
        this.user = null;
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.selectedFiles = [];
        this.jobsRefreshIntervalId = null;
        this.jobsAutoRefreshDelay = 5000;

        this.supportedFileTypes = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'application/vnd.ms-excel': '.xls',
            'text/html': '.html',
            'text/markdown': '.md',
            'text/csv': '.csv'
        };

        this.init();
    }

    async init() {
        console.log('RAGApp initializing...');
        // Compute WebSocket URL based on current location
        this.wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/updates';
        this.setupKeycloak();
        this.setupEventListeners();
    }

    setupKeycloak() {
        this.keycloak = new Keycloak({
            url: 'http://localhost:8080/',
            realm: 'rag_app',
            clientId: 'fastapi-client'
        });

        this.keycloak.init({ onLoad: 'check-sso' }).then(authenticated => {
            if (authenticated) {
                console.log('User is authenticated');
                // Merge id token and access token claims for completeness
                const idClaims = this.keycloak.idTokenParsed || {};
                const accessClaims = this.keycloak.tokenParsed || {};
                this.user = { ...idClaims, ...accessClaims };
                this.showMainApp();
                this.connectWebSocket();
            } else {
                console.log('User is not authenticated');
                this.showLogin();
            }
        }).catch(error => {
            console.error('Keycloak initialization failed:', error);
            this.showLogin();
        });

        this.keycloak.onTokenExpired = () => {
            this.keycloak.updateToken(30).catch(() => {
                console.error('Failed to refresh token');
                this.keycloak.logout();
            });
        };
    }

    setupEventListeners() {
        const loginBtn = document.getElementById('loginBtn');
        if (loginBtn) {
            loginBtn.addEventListener('click', () => this.keycloak.login());
        }

        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.keycloak.logout());
        }

        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabName = e.target.closest('.nav-tab').dataset.tab;
                this.switchTab(tabName);
            });
        });

        this.setupFileUpload();

        const submitQueryBtn = document.getElementById('submitQueryBtn');
        if (submitQueryBtn) {
            submitQueryBtn.addEventListener('click', this.handleQuery.bind(this));
        }

        const refreshDocuments = document.getElementById('refreshDocuments');
        if (refreshDocuments) {
            refreshDocuments.addEventListener('click', this.loadDocuments.bind(this));
        }

        const groupFilter = document.getElementById('groupFilter');
        const statusFilter = document.getElementById('statusFilter');
        if (groupFilter) groupFilter.addEventListener('change', this.loadDocuments.bind(this));
        if (statusFilter) statusFilter.addEventListener('change', this.loadDocuments.bind(this));
    }

    // ... (Keep all the other methods like setupFileUpload, handleFileSelection, etc., but update the API calls)

    async ensureFreshToken(minValiditySeconds = 30) {
        if (!this.keycloak) throw new Error('Keycloak not initialized');
        if (!this.keycloak.authenticated) throw new Error('User not authenticated');
        try {
            await this.keycloak.updateToken(minValiditySeconds);
        } catch (e) {
            console.error('Token refresh failed:', e);
            this.keycloak.logout();
            throw new Error('User not authenticated');
        }
    }

    async apiFetch(url, options = {}) {
        // Ensure token is fresh before making the request
        await this.ensureFreshToken(30);

        const makeRequest = async () => {
            const headers = {
                ...options.headers,
                'Authorization': `Bearer ${this.keycloak.token}`
            };
            return fetch(url, { ...options, headers });
        };

        let response = await makeRequest();

        // If unauthorized, try one refresh + retry
        if (response.status === 401) {
            try {
                await this.ensureFreshToken(30);
                response = await makeRequest();
            } catch (_) {
                // fallthrough to error handling
            }
        }

        if (!response.ok) {
            let error;
            try { error = await response.json(); } catch { error = {}; }
            throw new Error(error.detail || 'API request failed');
        }

        return response.json();
    }

    async handleFileUpload() {
        const groupSelect = document.getElementById('groupSelect');
        const groupId = groupSelect ? groupSelect.value : '';

        if (!groupId) {
            this.showToast('error', 'Error', 'Please select a group');
            return;
        }

        if (!this.selectedFiles || this.selectedFiles.length === 0) {
            this.showToast('error', 'Error', 'Please select files to upload');
            return;
        }

        const uploadProgress = document.getElementById('uploadProgress');
        const uploadBtn = document.getElementById('uploadBtn');

        if (uploadProgress) uploadProgress.style.display = 'block';
        if (uploadBtn) uploadBtn.disabled = true;

        try {
            const formData = new FormData();
            this.selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            formData.append('group_id', groupId);

            const result = await this.apiFetch(`${this.apiBase}/documents/upload`, {
                method: 'POST',
                body: formData
            });

            this.showToast('success', 'Upload Started',
                `Started processing ${result.files_count || this.selectedFiles.length} files. Monitor progress in the Jobs tab.`);
            this.resetUploadForm();
            this.loadJobs();
            this.switchTab('jobs');

        } catch (error) {
            console.error('Upload error:', error);
            this.showToast('error', 'Upload Failed', error.message);
            if (uploadProgress) uploadProgress.style.display = 'none';
            if (uploadBtn) uploadBtn.disabled = false;
        }
    }

    async handleQuery() {
        const queryInput = document.getElementById('queryInput');
        const queryText = queryInput ? queryInput.value.trim() : '';

        if (!queryText) {
            this.showToast('error', 'Error', 'Please enter a query');
            return;
        }

        const submitBtn = document.getElementById('submitQueryBtn');
        const queryResults = document.getElementById('queryResults');
        const queryResponse = document.getElementById('queryResponse');
        const queryStatus = document.getElementById('queryStatus');

        if (submitBtn) submitBtn.disabled = true;
        if (queryResults) queryResults.style.display = 'block';
        if (queryResponse) queryResponse.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i>Processing your query...</div>';
        if (queryStatus) queryStatus.innerHTML = '<i class="fas fa-clock"></i> Status: Processing';

        try {
            const result = await this.apiFetch(`${this.apiBase}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query_text: queryText })
            });

            this.showToast('success', 'Query Submitted', `Query submitted successfully. You will be notified upon completion.`);
            if (submitBtn) submitBtn.disabled = false;
            if (queryInput) queryInput.value = '';

        } catch (error) {
            console.error('Query error:', error);
            if (queryResponse) queryResponse.innerHTML = `<div class="error-message">Query failed: ${error.message}</div>`;
            if (queryStatus) queryStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Status: Failed';
            if (submitBtn) submitBtn.disabled = false;
        }
    }

    showLogin() {
        const loginForm = document.getElementById('loginForm');
        const mainApp = document.getElementById('mainApp');

        if (loginForm) loginForm.style.display = 'flex';
        if (mainApp) mainApp.style.display = 'none';
    }

    showMainApp() {
        const loginForm = document.getElementById('loginForm');
        const mainApp = document.getElementById('mainApp');
        const userInfo = document.getElementById('userInfo');
        const userName = document.getElementById('userName');

        if (loginForm) loginForm.style.display = 'none';
        if (mainApp) mainApp.style.display = 'block';
        if (userInfo) userInfo.style.display = 'flex';
        if (userName && this.user) userName.textContent = this.user.preferred_username;

        this.populateGroupSelects();
        this.loadDocuments();
        this.loadJobs();
    }

    populateGroupSelects() {
        if (!this.user) return;

        // Normalize groups to an array of strings
        const rawGroups = this.user.groups ?? this.user.group ?? null;
        const groups = Array.isArray(rawGroups)
            ? rawGroups
            : (rawGroups ? [rawGroups] : []);
        this.user.groups = groups;

        const groupSelect = document.getElementById('groupSelect');
        const groupFilter = document.getElementById('groupFilter');

        const personalOption = document.createElement('option');
        personalOption.value = this.user.sub; // Use user's subject ID for personal group
        personalOption.textContent = 'Personal';

        if (groupSelect) {
            groupSelect.innerHTML = '<option value="">Select a group...</option>';
            groupSelect.appendChild(personalOption.cloneNode(true));
            this.user.groups.forEach(group => {
                const option = document.createElement('option');
                option.value = group;
                option.textContent = group;
                groupSelect.appendChild(option);
            });
        }

        if (groupFilter) {
            groupFilter.innerHTML = '<option value="">All Groups</option>';
            groupFilter.appendChild(personalOption.cloneNode(true));
            this.user.groups.forEach(group => {
                const option = document.createElement('option');
                option.value = group;
                option.textContent = group;
                groupFilter.appendChild(option);
            });
        }
    }

    async loadDocuments() {
        const documentsGrid = document.getElementById('documentsGrid');
        const documentsLoading = document.getElementById('documentsLoading');
        const groupFilter = document.getElementById('groupFilter');
        const statusFilter = document.getElementById('statusFilter');

        if (documentsLoading) documentsLoading.style.display = 'flex';
        if (documentsGrid) documentsGrid.innerHTML = '';

        try {
            const params = new URLSearchParams();
            if (groupFilter && groupFilter.value) params.append('group_id', groupFilter.value);
            if (statusFilter && statusFilter.value) params.append('status', statusFilter.value);
            params.append('limit', '50');

            const result = await this.apiFetch(`${this.apiBase}/documents?${params}`);
            this.displayDocuments(result.documents || []);

        } catch (error) {
            console.error('Error loading documents:', error);
            if (documentsGrid) {
                documentsGrid.innerHTML = `<div class="error-message">Failed to load documents: ${error.message}</div>`;
            }
        } finally {
            if (documentsLoading) documentsLoading.style.display = 'none';
        }
    }

    async deleteDocument(documentId) {
        if (!confirm('Are you sure you want to delete this document?')) {
            return;
        }

        try {
            await this.apiFetch(`${this.apiBase}/documents/${documentId}`, { method: 'DELETE' });
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

        if (jobsLoading) jobsLoading.style.display = 'flex';
        if (jobsList) jobsList.innerHTML = '';

        try {
            const result = await this.apiFetch(`${this.apiBase}/jobs`);
            const jobs = result.jobs || [];
            this.displayJobs(jobs);

            const stats = this.calculateJobStats(jobs);
            if (activeJobs) activeJobs.textContent = stats.active;
            if (completedJobs) completedJobs.textContent = stats.completed;
            if (failedJobs) failedJobs.textContent = stats.failed;

        } catch (error) {
            console.error('Error loading jobs:', error);
            if (jobsList) {
                jobsList.innerHTML = `<div class="error-message">Failed to load jobs: ${error.message}</div>`;
            }
        } finally {
            if (jobsLoading) jobsLoading.style.display = 'none';
        }
    }

    async cancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) {
            return;
        }

        try {
            await this.apiFetch(`${this.apiBase}/jobs/${jobId}`, { method: 'DELETE' });
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

        // Ensure token is refreshed before opening WS
        this.ensureFreshToken(30)
            .then(() => {
                try {
                    const wsUrlWithToken = `${this.wsUrl}?token=${encodeURIComponent(this.keycloak.token)}`;
                    this.websocket = new WebSocket(wsUrlWithToken);

                    this.websocket.onopen = () => {
                        console.log('WebSocket connected');
                        this.reconnectAttempts = 0;
                        this.updateConnectionStatus('connected');
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
            })
            .catch(() => {
                // Not authenticated; ensure UI reflects it
                this.updateConnectionStatus('disconnected');
                this.showLogin();
            });
    }

    // ... (Keep all other methods like displayDocuments, displayJobs, handleWebSocketMessage, etc. as they are)
    // Make sure to copy the remaining methods from the old file here.

    // NOTE: The following methods are copied from the old file and should be kept.
    setupFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');

        if (!uploadArea || !fileInput || !uploadBtn) return;

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
                this.isSupportedFileType(file)
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

        uploadBtn.addEventListener('click', (e) => {
            this.handleFileUpload();
        });
    }

    handleFileSelection(files) {
        if (files.length === 0) return;

        const fileList = document.getElementById('fileList');
        const selectedFiles = document.getElementById('selectedFiles');

        if (!fileList || !selectedFiles) return;

        selectedFiles.innerHTML = '';
        this.selectedFiles = files;

        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="file-info">
                    ${this.getFileIcon(file.name)}
                    <div class="file-details">
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${this.formatFileSize(file.size)}</div>
                        <div class="file-type">${this.getFileTypeLabel(file.name)}</div>
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

    resetUploadForm() {
        const fileList = document.getElementById('fileList');
        const uploadProgress = document.getElementById('uploadProgress');
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');

        if (fileList) fileList.style.display = 'none';
        if (uploadProgress) uploadProgress.style.display = 'none';
        if (fileInput) fileInput.value = '';
        if (uploadBtn) uploadBtn.disabled = false;
        this.selectedFiles = [];
    }

    switchTab(tabName) {
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeTab) activeTab.classList.add('active');

        document.querySelectorAll('.tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        const activePane = document.getElementById(`${tabName}Tab`);
        if (activePane) activePane.classList.add('active');

        if (tabName === 'documents') {
            this.loadDocuments();
        } else if (tabName === 'jobs') {
            this.loadJobs();
            this.startJobsAutoRefresh();
        } else {
            this.stopJobsAutoRefresh();
        }
    }

    displayDocuments(documents) {
        const documentsGrid = document.getElementById('documentsGrid');
        if (!documentsGrid) return;

        if (documents.length === 0) {
            documentsGrid.innerHTML = '<div class="loading">No documents found</div>';
            return;
        }

        documentsGrid.innerHTML = documents.map(doc => `
            <div class="document-card">
                <div class="document-header">
                    ${this.getFileIcon(doc.filename || 'unknown.pdf').replace('file-icon', 'document-icon')}
                    <div class="document-title">${doc.filename || 'Unknown'}</div>
                </div>
                <div class="document-meta">
                    <div class="document-meta-item">
                        <span>Group:</span>
                        <span>${doc.group_id || 'Unknown'}</span>
                    </div>
                    <div class="document-meta-item">
                        <span>Size:</span>
                        <span>${this.formatFileSize(doc.file_size || 0)}</span>
                    </div>
                    <div class="document-meta-item">
                        <span>Uploaded:</span>
                        <span>${this.formatDate(doc.upload_date || new Date())}</span>
                    </div>
                    <div class="document-meta-item">
                        <span>Status:</span>
                        <span class="document-status status-${doc.processing_status || 'unknown'}">
                            ${this.getStatusIcon(doc.processing_status || 'unknown')}
                            ${doc.processing_status || 'unknown'}
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
        if (!jobsList) return;

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
                        <div class="job-progress-fill" style="width: ${(job.progress || 0) * 100}%"></div>
                    </div>
                    <div class="job-progress-text">${Math.round((job.progress || 0) * 100)}% complete</div>
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

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts && this.keycloak.authenticated) {
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
            case 'pong':
                console.log('Received pong from server');
                break;
            case 'job_notification': {
                if (message.job) {
                    this.handleJobUpdate({
                        job_id: message.job.job_id,
                        status: message.job.status,
                        progress: message.job.progress
                    });
                }
                break;
            }
            case 'job_progress': {
                this.handleJobUpdate({
                    job_id: message.job_id,
                    status: 'processing',
                    progress: message.progress
                });
                break;
            }
            case 'job_update':
                this.handleJobUpdate(message.data);
                break;
            case 'notification':
                this.showToast(message.data.level, message.data.title, message.data.message);
                break;
            case 'query_result':
                this.handleQueryResult(message.data);
                break;
            case 'error':
                console.error('WebSocket error:', message.error);
                this.showToast('error', 'Connection Error', message.error.message);
                break;
            case 'connection_established':
                console.log('WebSocket connection established:', message);
                break;
            default:
                console.log('Unknown WebSocket message type:', message.type);
        }
    }

    handleJobUpdate(jobData) {
        const jobCard = document.querySelector(`[data-job-id="${jobData.job_id}"]`);
        if (jobCard) {
            const progressFill = jobCard.querySelector('.job-progress-fill');
            const progressText = jobCard.querySelector('.job-progress-text');
            const statusElement = jobCard.querySelector('.document-status');

            if (progressFill) {
                progressFill.style.width = `${(jobData.progress || 0) * 100}%`;
            }
            if (progressText) {
                progressText.textContent = `${Math.round((jobData.progress || 0) * 100)}% complete`;
            }
            if (statusElement) {
                statusElement.className = `document-status status-${jobData.status}`;
                statusElement.innerHTML = `${this.getStatusIcon(jobData.status)} ${jobData.status}`;
            }
        }

        if (jobData.status === 'completed' || jobData.status === 'failed') {
            this.loadDocuments();
            this.loadJobs();
            setTimeout(() => this.maybeStopJobsAutoRefresh(), 0);
        }
    }

    startJobsAutoRefresh() {
        this.stopJobsAutoRefresh();
        this.jobsRefreshIntervalId = setInterval(async () => {
            try {
                await this.loadJobs();
                this.maybeStopJobsAutoRefresh();
            } catch (_) {}
        }, this.jobsAutoRefreshDelay);
    }

    stopJobsAutoRefresh() {
        if (this.jobsRefreshIntervalId) {
            clearInterval(this.jobsRefreshIntervalId);
            this.jobsRefreshIntervalId = null;
        }
    }

    maybeStopJobsAutoRefresh() {
        const jobsList = document.getElementById('jobsList');
        if (!jobsList) return;
        const hasActive = Array.from(jobsList.querySelectorAll('.job-card .document-status')).some(el => {
            return el.classList.contains('status-processing') || el.classList.contains('status-pending');
        });
        if (!hasActive) this.stopJobsAutoRefresh();
    }

    handleQueryResult(data) {
        const queryResponse = document.getElementById('queryResponse');
        const queryStatus = document.getElementById('queryStatus');

        if (data.status === 'completed') {
            if (queryResponse) queryResponse.innerHTML = data.result || 'No results found';
            if (queryStatus) queryStatus.innerHTML = '<i class="fas fa-check-circle"></i> Status: Completed';
        } else if (data.status === 'failed') {
            if (queryResponse) queryResponse.innerHTML = `<div class="error-message">Query failed: ${data.error || 'Unknown error'}</div>`;
            if (queryStatus) queryStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> Status: Failed';
        }
    }

    startHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }

        this.heartbeatInterval = setInterval(() => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    updateConnectionStatus(status) {
        const connectionStatus = document.getElementById('connectionStatus');
        const connectionText = document.getElementById('connectionText');

        if (connectionStatus && connectionText) {
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
                default:
                    connectionText.textContent = 'Unknown';
            }
        }
    }

    showToast(level, title, message) {
        const toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            console.log(`${level.toUpperCase()}: ${title} - ${message}`);
            if (level === 'error') {
                alert(`Error: ${message}`);
            }
            return;
        }

        const toast = document.createElement('div');
        toast.className = `toast toast-${level}`;
        toast.innerHTML = `
            <div class="toast-header">
                <strong>${title}</strong>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="toast-body">${message}</div>
        `;

        toastContainer.appendChild(toast);

        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDate(dateString) {
        try {
            const date = new Date(dateString);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        } catch (error) {
            return 'Invalid Date';
        }
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

    isSupportedFileType(file) {
        if (this.supportedFileTypes[file.type]) {
            return true;
        }
        
        const fileName = file.name.toLowerCase();
        const supportedExtensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.xls', '.html', '.md', '.csv'];
        return supportedExtensions.some(ext => fileName.endsWith(ext));
    }

    getFileIcon(fileName) {
        const ext = fileName.toLowerCase().split('.').pop();
        switch (ext) {
            case 'pdf':
                return '<i class="fas fa-file-pdf file-icon"></i>';
            case 'docx':
            case 'doc':
                return '<i class="fas fa-file-word file-icon"></i>';
            case 'pptx':
            case 'ppt':
                return '<i class="fas fa-file-powerpoint file-icon"></i>';
            case 'xlsx':
            case 'xls':
                return '<i class="fas fa-file-excel file-icon"></i>';
            case 'html':
            case 'htm':
                return '<i class="fas fa-file-code file-icon"></i>';
            case 'md':
                return '<i class="fab fa-markdown file-icon"></i>';
            case 'csv':
                return '<i class="fas fa-file-csv file-icon"></i>';
            default:
                return '<i class="fas fa-file file-icon"></i>';
        }
    }

    getFileTypeLabel(fileName) {
        const ext = fileName.toLowerCase().split('.').pop();
        switch (ext) {
            case 'pdf':
                return 'PDF Document';
            case 'docx':
            case 'doc':
                return 'Word Document';
            case 'pptx':
            case 'ppt':
                return 'PowerPoint Presentation';
            case 'xlsx':
            case 'xls':
                return 'Excel Spreadsheet';
            case 'html':
            case 'htm':
                return 'HTML Document';
            case 'md':
                return 'Markdown Document';
            case 'csv':
                return 'CSV Spreadsheet';
            default:
                return 'Document';
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new RAGApp();
});