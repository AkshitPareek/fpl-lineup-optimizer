import { Card } from '../components/ui'

/** Two-column layout: a sticky settings panel beside the results canvas. */
export function Workspace({ settings, children }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)] xl:grid-cols-[320px_minmax(0,1fr)] gap-6 items-start">
      <aside className="lg:sticky lg:top-24">
        <Card className="p-5 space-y-5">{settings}</Card>
      </aside>
      <div className="min-w-0 animate-fade-in">{children}</div>
    </div>
  )
}

export function Canvas({ children, className = '' }) {
  return <Card className={`min-h-[360px] lg:min-h-[560px] flex flex-col p-4 sm:p-6 ${className}`}>{children}</Card>
}
