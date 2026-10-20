import { fetchModules } from '../lib/api'
import useAsync from '../lib/useAsync'
import ModuleCard from './ModuleCard'
import { Button, ErrorBox } from './ui'

export default function ModuleGrid() {
  const { data, error, loading, reload } = useAsync('modules', fetchModules)

  if (error)
    return (
      <ErrorBox title="Could not load modules" action={<Button tone="ghost" onClick={reload}>Retry</Button>}>
        {error.message}
      </ErrorBox>
    )

  if (loading || !data)
    return (
      <div className="grid gap-5 sm:grid-cols-2">
        {[0, 1].map((i) => (
          <div key={i} className="glass h-72 animate-pulse rounded-2xl" />
        ))}
      </div>
    )

  const modules = [...data].sort(
    (a, b) => (a.status === 'ready' ? 0 : 1) - (b.status === 'ready' ? 0 : 1) || (a.number ?? 99) - (b.number ?? 99),
  )
  return (
    <div className="grid gap-5 sm:grid-cols-2">
      {modules.map((m, i) => (
        <ModuleCard key={m.id} module={m} index={i} />
      ))}
    </div>
  )
}
