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
            staleTime: 1000 * 60 * 5, // 5 minutes
            refetchOnWindowFocus: false,
            refetchOnMount: false, // Manual refetch only
            refetchOnReconnect: false,
            refetchInterval: false, // No automatic refetching
            retry: 1, // Only retry once
            networkMode: 'online',
            gcTime: 1000 * 60 * 30, // 30 minutes
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
