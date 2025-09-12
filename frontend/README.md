# RAG System Frontend

A production-ready Next.js frontend for the multi-user RAG system, built with TypeScript, Tailwind CSS, and modern React patterns.

## Features

- 🚀 **Modern Stack**: Next.js 14 with App Router, TypeScript, Tailwind CSS
- 🎨 **Beautiful UI**: shadcn/ui components with Radix primitives
- ⚡ **Smooth Animations**: Framer Motion with accessibility support
- 📱 **Responsive Design**: Mobile-first approach with responsive layouts
- 🔄 **Smart Caching**: TanStack Query for data fetching and caching
- ♿ **Accessibility**: Full keyboard navigation, ARIA labels, focus management
- 🎯 **Type Safe**: Full TypeScript coverage with strict mode

## Pages

### 🏠 Home Page (`/`)
- Landing page with feature overview
- Navigation cards to main sections
- Animated hero section

### 📤 Upload Documents (`/upload`)
- **File Upload**: Drag & drop interface for PDF, DOCX, PPTX, XLSX, HTML, MD, CSV
- **URL Processing**: Web page content processing (placeholder)
- **Text Processing**: Raw text content processing (placeholder)
- Real-time file validation and preview
- Dataset/group selection

### 🔍 Query Documents (`/query`)
- Interactive query interface
- Dataset selector with top_k configuration
- Real-time results with score badges
- Source preview drawers
- Query history integration

### 📄 My Documents (`/my-documents`)
- Document library management
- Search and filtering capabilities
- Bulk operations with confirmations
- Document preview and metadata

### 🎯 Job Status (`/job-status`)
- Real-time job monitoring
- Paginated job history
- Live progress updates
- Job cancellation and management

### 📚 Query History (`/query-history`)
- Historical query browser
- Search through past queries
- Re-run capability
- Detailed result viewing

## Quick Start

### Prerequisites

- Node.js 18+ and npm 8+
- Backend API running on port 8000 (or configured URL)

### Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.local.example .env.local

# Edit .env.local with your API URL
# NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`.

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Backend Integration

The frontend integrates with your FastAPI backend through the `lib/api.ts` client. The API client automatically:

- **Handles Authentication**: JWT token management (ready for Keycloak)
- **Retry Logic**: Automatic retry for GET requests with exponential backoff
- **Error Handling**: Consistent error formatting and user feedback
- **Type Safety**: Full TypeScript interfaces matching your backend models

### API Endpoints Used

- `GET /api/documents` - List user documents
- `POST /api/documents/upload` - Upload files for processing
- `DELETE /api/documents/{id}` - Delete documents
- `GET /api/jobs` - List user jobs
- `POST /api/jobs/{id}/cancel` - Cancel jobs
- `POST /api/query` - Submit queries
- `GET /api/query/{id}` - Get query results
- `GET /api/query-history/user` - Get query history

## Architecture

### State Management
- **TanStack Query** for server state and caching
- **Local State** with React hooks for UI state
- **Optimistic Updates** for better UX

### Component Structure
```
components/
├── ui/                 # shadcn/ui base components
├── AnimatedDrawer.tsx  # Reusable drawer component
├── DataTable.tsx       # Generic data table
├── TaskStatus.tsx      # Job/task status display
└── Skeletons.tsx       # Loading states
```

### Utilities
```
lib/
├── api.ts             # Backend API client
├── utils.ts           # Utility functions
├── highlight.ts       # Text highlighting
└── markdown.tsx       # Markdown rendering
```

### Styling
- **Tailwind CSS** for utility-first styling
- **CSS Variables** for theming
- **Custom Animations** with reduced motion support
- **Responsive Design** with mobile-first approach

## Key Features

### File Upload
- **Drag & Drop**: Native HTML5 drag and drop
- **File Validation**: Type and size checking
- **Progress Tracking**: Real-time upload progress
- **Error Handling**: User-friendly error messages

### Query Interface
- **Live Search**: Real-time query processing
- **Result Highlighting**: Search term highlighting
- **Source Preview**: Modal document viewing
- **History Integration**: Save and replay queries

### Job Monitoring
- **Real-time Updates**: Live job status updates
- **Progress Indicators**: Visual progress bars
- **Bulk Operations**: Batch job management
- **Error Recovery**: Retry failed operations

### Accessibility
- **Keyboard Navigation**: Full keyboard support
- **Screen Readers**: ARIA labels and descriptions
- **Focus Management**: Logical focus flow
- **Reduced Motion**: Respects user preferences

## Development

### Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # TypeScript type checking
```

### Environment Variables

```bash
# Required
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Optional
NEXT_PUBLIC_USE_MOCK_API=false  # Enable mock API for offline dev
```

### Customization

The frontend is designed to be easily customizable:

1. **API Endpoints**: Modify `lib/api.ts` to match your backend URLs
2. **Styling**: Update Tailwind config and CSS variables
3. **Components**: Extend or modify shadcn/ui components
4. **Features**: Add new pages following the established patterns

## Performance

### Optimizations
- **Code Splitting**: Automatic route-based code splitting
- **Image Optimization**: Next.js Image component
- **Caching**: TanStack Query with smart cache invalidation
- **Bundle Analysis**: Built-in bundle analyzer

### Monitoring
- **Web Vitals**: Core Web Vitals tracking ready
- **Error Boundaries**: React error boundaries for fault tolerance
- **Loading States**: Comprehensive loading and skeleton states

## Browser Support

- **Modern Browsers**: Chrome 90+, Firefox 90+, Safari 14+, Edge 90+
- **Mobile**: iOS Safari 14+, Chrome Mobile 90+
- **Accessibility**: WCAG 2.1 AA compliant

## Contributing

The codebase follows these patterns:

1. **TypeScript**: Strict mode with comprehensive types
2. **Component Design**: Composition over inheritance
3. **Accessibility First**: Every component is accessible by default
4. **Performance**: Optimistic updates and smart caching
5. **Testing Ready**: Components designed for easy testing

## Deployment

The frontend can be deployed to any static hosting provider:

- **Vercel**: Optimal for Next.js (recommended)
- **Netlify**: Static export with API proxy
- **AWS S3 + CloudFront**: Static hosting with CDN
- **Docker**: Containerized deployment

Make sure to configure the `NEXT_PUBLIC_API_BASE_URL` environment variable for your production API.
