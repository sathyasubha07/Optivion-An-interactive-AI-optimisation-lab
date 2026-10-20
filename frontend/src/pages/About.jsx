import { fetchContributors } from '../lib/api'
import useAsync from '../lib/useAsync'
import { Button, ErrorBox, FadeIn, Glass, SectionTitle } from '../components/ui'

const stack = [
  ['Backend', 'FastAPI + uvicorn. One adapter per module calls that module’s own entry point.'],
  ['Frontend', 'React, Vite, Tailwind CSS, Framer Motion, Recharts, KaTeX.'],
  ['Jobs', 'Runs execute in a background worker; the UI polls job status.'],
]

export default function About() {
  const { data, error, loading, reload } = useAsync('contributors', fetchContributors)

  return (
    <section className="mx-auto max-w-5xl px-4 py-16 sm:px-6">
      <FadeIn>
        <SectionTitle eyebrow="About" title="OptiVision">
          An interactive lab that makes optimisation visible. A group project for the MFC course by Team 08.
          Each module is integrated through an adapter, without modifying its source.
        </SectionTitle>
      </FadeIn>

      <FadeIn delay={0.05}>
        <Glass className="mt-10 overflow-hidden">
          <div className="border-b border-white/5 px-6 py-4">
            <h3 className="font-display text-lg font-semibold">Modules</h3>
          </div>
          {error && (
            <div className="p-6">
              <ErrorBox title="Could not load credits" action={<Button tone="ghost" onClick={reload}>Retry</Button>}>
                {error.message}
              </ErrorBox>
            </div>
          )}
          {loading && <div className="h-48 animate-pulse" />}
          {data && (
            <div className="scroll-thin overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase tracking-wider text-mist-500">
                  <tr>
                    <th className="px-6 py-3 font-medium">Module</th>
                    <th className="px-6 py-3 font-medium">Method</th>
                  </tr>
                </thead>
                <tbody>
                  {data.table.map((row) => (
                    <tr key={row.module} className="border-t border-white/5">
                      <td className="px-6 py-4 font-medium text-mist-100">
                        <span className="mr-2 font-mono text-mist-500">{row.number}</span>
                        {row.module}
                      </td>
                      <td className="px-6 py-4 text-mist-300">{row.method}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Glass>
      </FadeIn>

      <div className="mt-8 grid gap-5 md:grid-cols-3">
        {stack.map(([t, b], i) => (
          <FadeIn key={t} delay={0.05 * i}>
            <Glass className="h-full p-6">
              <h3 className="font-display font-semibold">{t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-mist-300">{b}</p>
            </Glass>
          </FadeIn>
        ))}
      </div>

      <FadeIn>
        <Glass className="mt-8 p-6 sm:p-8">
          <h3 className="font-display text-lg font-semibold">Honesty rules</h3>
          <ul className="mt-3 space-y-2 text-sm leading-relaxed text-mist-300">
            <li>• The interface shows only values returned by the backend for your run. Nothing is hard-coded.</li>
            <li>• When a value is unavailable it is shown as “not computed”, never estimated.</li>
            <li>• Iteration views are replays of what the solver recorded, and are labelled as such.</li>
            <li>• Sample datasets are labelled. Synthetic data is never presented as a benchmark result.</li>
          </ul>
          {data?.github && (
            <div className="mt-6">
              <Button tone="ghost" href={data.github} target="_blank" rel="noreferrer">
                View the repository ↗
              </Button>
            </div>
          )}
        </Glass>
      </FadeIn>
    </section>
  )
}
