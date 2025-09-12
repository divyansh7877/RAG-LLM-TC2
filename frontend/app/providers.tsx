'use client'

import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { AuthProvider } from '@/lib/auth'
import { AuthenticatedApiProvider } from '@/lib/authenticated-api'

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 1000 * 60 * 10, // 10 minutes
            refetchOnWindowFocus: false,
            refetchOnMount: true,
            refetchOnReconnect: false,
            refetchInterval: false, // Disable automatic refetching
            retry: false, // Disable retries completely to prevent loops
            networkMode: 'online', // Only run queries when online
            gcTime: 1000 * 60 * 30, // 30 minutes garbage collection
          },
        },
      })
  )

  return (
    <AuthProvider>
      <AuthenticatedApiProvider>
        <QueryClientProvider client={queryClient}>
          {children}
          <Toaster position="top-right" />
        </QueryClientProvider>
      </AuthenticatedApiProvider>
    </AuthProvider>
  )
}
