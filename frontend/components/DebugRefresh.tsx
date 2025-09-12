'use client'

import { useEffect, useRef } from 'react'

export function DebugRefresh({ name }: { name: string }) {
  const renderCount = useRef(0)
  const lastRender = useRef(Date.now())
  
  useEffect(() => {
    renderCount.current++
    const now = Date.now()
    const timeSinceLastRender = now - lastRender.current
    lastRender.current = now
    
    console.log(`[${name}] Render #${renderCount.current} (${timeSinceLastRender}ms since last)`)
  })
  
  return (
    <div className="fixed top-0 right-0 bg-red-500 text-white p-2 text-xs z-50 opacity-75">
      {name}: {renderCount.current}
    </div>
  )
}