import { AnimatePresence, motion } from 'framer-motion'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import Tex from '../components/Tex'
import { BarChartCard, DataTable, Downloads, Notes } from '../components/results/Blocks'
import { CompareSlider, ImageGrid } from '../components/results/Images'
import { MetricGroups, SummaryCards } from '../components/results/Metrics'
import Replay from '../components/results/Replay'
import { Button, Chip, ErrorBox, Eyebrow, Glass, ModuleGlyph, StatusPill } from '../components/ui'
import { fetchJob, fetchModule, startRun } from '../lib/api'
import useAsync from '../lib/useAsync'
import RunPanel from '../workspace/RunPanel'

const STEPS = [
  { key: 'problem', label: 'Problem' },
  { key: 'why', label: 'Why this method' },
  { key: 'math', label: 'Mathematics' },
  { key: 'run', label: 'Upload & run' },
  { key: 'replay', label: 'Execution replay', needsResult: true },
  { key: 'results', label: 'Results', needsResult: true },
  { key: 'explain', label: 'Explanation', needsResult: true },
]

const IDLE = { phase: 'idle' }
const POLL_MS = 700

function H({ children, sub }) {
  return (
    <div className="mb-5">
      <h3 className="font-display text-2xl font-semibold text-mist-100">{children}</h3>
      {sub && <p className="mt-1 text-sm text-mist-300">{sub}</p>}
    </div>
  )
}

function ProblemStep({ module }) {
  return (
    <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
      <Glass className="p-6">
        <H>The problem</H>
        <p className="leading-relaxed text-mist-300">{module.problem}</p>
        <div className="mt-6 rounded-xl border border-white/5 bg-white/[0.03] p-4">
          <p className="text-xs uppercase tracking-wider text-mist-500">Your input</p>
          <p className="mt-1 text-sm text-mist-300">{module.input_help}</p>
        </div>
      </Glass>
      <Glass className="p-6">
        <H>What the pipeline does</H>
        <ol className="space-y-3">
          {module.algorithm_steps?.map((s, i) => (
            <li key={i} className="flex gap-3 text-sm leading-relaxed text-mist-300">
              <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-white/5 font-mono text-[11px] text-teal">
                {i + 1}
              </span>
              <span>{s}</span>
            </li>
          ))}
        </ol>
      </Glass>
    </div>
  )
}

function WhyStep({ module }) {
  return (
    <div className="space-y-5">
      <Glass className="p-6">
        <H>Why {module.method_chips?.[0] ?? 'this method'}?</H>
        <p className="max-w-3xl leading-relaxed text-mist-300">{module.why_this_method}</p>
      </Glass>
      {module.alternatives_compared?.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2">
          {module.alternatives_compared.map((a) => (
            <Glass key={a.name} className="p-5">
              <p className="font-display font-semibold text-mist-100">{a.name}</p>
              <p className="mt-1 text-sm text-mist-300">{a.what}</p>
              <p className="mt-3 text-sm text-amber/90">
                <span className="font-semibold">Why not: </span>
                {a.why_not}
              </p>
            </Glass>
          ))}
        </div>
      )}
    </div>
  )
}

function MathStep({ module }) {
  const m = module.math ?? {}
  return (
    <div className="space-y-5">
      <Glass className="p-6 sm:p-8">
        <Eyebrow>Objective</Eyebrow>
        <div className="mt-4 text-lg text-mist-100 sm:text-xl">
          <Tex display>{m.objective}</Tex>
        </div>
        {m.constraints?.length > 0 && (
          <div className="mt-6">
            <Eyebrow>Subject to</Eyebrow>
            <div className="mt-3 flex flex-wrap gap-x-8 gap-y-2 text-mist-100">
              {m.constraints.map((c) => (
                <Tex key={c}>{c}</Tex>
              ))}
            </div>
          </div>
        )}
        {m.update && (
          <div className="mt-6 rounded-xl border border-white/5 bg-white/[0.03] p-4">
            <Eyebrow>{m.update.label}</Eyebrow>
            <div className="mt-3 text-mist-100">
              <Tex display>{m.update.latex}</Tex>
            </div>
          </div>
        )}
        {m.dimensions && <p className="mt-5 text-sm text-mist-500">{m.dimensions}</p>}
      </Glass>
      {m.symbols?.length > 0 && (
        <Glass className="p-6">
          <H>Symbols</H>
          <dl className="grid gap-x-8 gap-y-4 sm:grid-cols-2">
            {m.symbols.map((s) => (
              <div key={s.symbol} className="flex gap-4">
                <dt className="w-24 shrink-0 text-mist-100">
                  <Tex>{s.symbol}</Tex>
                </dt>
                <dd className="text-sm leading-relaxed text-mist-300">{s.meaning}</dd>
              </div>
            ))}
          </dl>
        </Glass>
      )}
    </div>
  )
}

function ResultsStep({ result }) {
  const images = result.images ?? []
  const byKey = Object.fromEntries(images.map((i) => [i.key, i]))
  const compare = result.compare && byKey[result.compare.before] && byKey[result.compare.after]
  return (
    <div className="space-y-8">
      <SummaryCards metrics={result.summary} />
      {(compare || images.length > 0) && (
        <section>
          <H>Images</H>
          <div className="space-y-5">
            {compare && (
              <CompareSlider
                before={byKey[result.compare.before]}
                after={byKey[result.compare.after]}
                label={result.compare.label}
              />
            )}
            <ImageGrid items={images} />
          </div>
        </section>
      )}
      {result.metric_groups?.length > 0 && (
        <section>
          <H>All metrics</H>
          <MetricGroups groups={result.metric_groups} />
        </section>
      )}
      {result.charts?.length > 0 && (
        <section className="space-y-5">
          {result.charts.map((c) => (c.type === 'bar' ? <BarChartCard key={c.key} chart={c} /> : null))}
        </section>
      )}
      {result.tables?.length > 0 && (
        <section className="grid gap-5">
          {result.tables.map((t) => (
            <DataTable key={t.key} table={t} />
          ))}
        </section>
      )}
      {result.plots?.some((p) => p.url) && (
        <section>
          <H sub="Generated by the module itself for this run. Click to enlarge.">Module plots</H>
          <ImageGrid items={result.plots} columns="sm:grid-cols-2" light />
        </section>
      )}
      <Notes notes={result.notes} />
    </div>
  )
}

function ExplainStep({ result }) {
  return (
    <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
      <Glass className="p-5 sm:p-6">
        <H sub={result.explanation_source}>Explanation</H>
        <pre className="scroll-thin max-h-[640px] overflow-auto whitespace-pre-wrap rounded-xl bg-black/30 p-4 font-mono text-[12.5px] leading-relaxed text-mist-300">
          {result.explanation}
        </pre>
      </Glass>
      <div>
        <Glass className="p-5 sm:p-6">
          <H sub="Files written by this run on the backend.">Downloads</H>
          <Downloads downloads={result.downloads} />
        </Glass>
      </div>
    </div>
  )
}

function Stepper({ step, setStep, hasResult }) {
  return (
    <div className="scroll-thin -mx-4 overflow-x-auto px-4 sm:mx-0 sm:px-0">
      <ol className="flex min-w-max gap-2">
        {STEPS.map((s, i) => {
          const locked = s.needsResult && !hasResult
          const active = i === step
          return (
            <li key={s.key}>
              <button
                type="button"
                disabled={locked}
                onClick={() => setStep(i)}
                aria-current={active ? 'step' : undefined}
                title={locked ? 'Run the module first' : undefined}
                className={`relative flex items-center gap-2 rounded-xl px-3.5 py-2 text-sm transition ${
                  active ? 'text-ink-950' : locked ? 'cursor-not-allowed text-mist-500/60' : 'text-mist-300 hover:bg-white/5 hover:text-mist-100'
                }`}
              >
                {active && (
                  <motion.span layoutId="step-pill" className="bg-gradient-accent absolute inset-0 rounded-xl" transition={{ type: 'spring', bounce: 0.2, duration: 0.5 }} />
                )}
                <span className="relative font-mono text-xs">{String(i + 1).padStart(2, '0')}</span>
                <span className="relative font-medium">{s.label}</span>
                {locked && <span className="relative text-[10px]">🔒</span>}
              </button>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

// Remount per module id so step/run state never leaks between modules.
export default function WorkspaceRoute() {
  const { id } = useParams()
  return <Workspace key={id} id={id} />
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

function Workspace({ id }) {
  const { data: module, error, loading, reload } = useAsync(`module:${id}`, () => fetchModule(id))
  const [step, setStep] = useState(0)
  const [run, setRun] = useState(IDLE)
  const [result, setResult] = useState(null)
  const runToken = useRef(0)
  const topRef = useRef(null)

  useEffect(
    () => () => {
      runToken.current += 1 // stop polling on unmount
    },
    [],
  )

  const goto = useCallback((i) => {
    setStep(i)
    topRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  const onRun = useCallback(
    async ({ file, sampleId }) => {
      const token = ++runToken.current
      const live = () => token === runToken.current
      const inputLabel = file ? file.name : `${module.samples.find((s) => s.id === sampleId)?.label ?? sampleId} (sample)`
      setRun({ phase: 'submitting', inputLabel })
      let jobId
      try {
        ;({ job_id: jobId } = await startRun(module.id, { file, sampleId }))
      } catch (e) {
        if (live()) setRun({ phase: 'failed', stage: 'submit', error: e.message, inputLabel })
        return
      }
      while (live()) {
        let job
        try {
          job = await fetchJob(jobId)
        } catch (e) {
          if (live()) setRun({ phase: 'failed', stage: 'job', error: e.message, inputLabel })
          return
        }
        if (!live()) return
        if (job.status === 'succeeded') {
          setResult(job.result)
          setRun({ phase: 'succeeded', inputLabel, elapsed: job.elapsed_seconds })
          await sleep(600)
          if (live()) goto(STEPS.findIndex((s) => s.key === 'replay'))
          return
        }
        if (job.status === 'failed') {
          setRun({ phase: 'failed', stage: 'job', error: job.message, inputLabel })
          return
        }
        setRun({ phase: job.status, inputLabel, elapsed: job.elapsed_seconds })
        await sleep(POLL_MS)
      }
    },
    [module, goto],
  )

  if (error)
    return (
      <section className="mx-auto max-w-3xl px-4 py-20 sm:px-6">
        <ErrorBox
          title={error.status === 404 ? 'Module not found' : 'Could not load this module'}
          action={
            <div className="flex gap-3">
              <Button tone="ghost" onClick={reload}>
                Retry
              </Button>
              <Button tone="subtle" to="/modules">
                All modules
              </Button>
            </div>
          }
        >
          {error.message}
        </ErrorBox>
      </section>
    )

  if (loading || !module)
    return (
      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
        <div className="glass h-40 animate-pulse rounded-2xl" />
      </section>
    )

  if (module.status !== 'ready')
    return (
      <section className="mx-auto max-w-3xl px-4 py-20 text-center sm:px-6">
        <ModuleGlyph module={module} size="mx-auto h-14 w-14" />
        <h1 className="mt-6 font-display text-3xl font-semibold">{module.name}</h1>
        <p className="mt-3 text-mist-300">{module.short_description}</p>
        <p className="mt-2 text-sm text-mist-500">Planned method: {module.method}</p>
        <div className="mt-6 flex justify-center">
          <StatusPill status={module.status} />
        </div>
        <Button tone="ghost" to="/modules" className="mt-8">
          ← All modules
        </Button>
      </section>
    )

  const hasResult = Boolean(result)
  const current = STEPS[step]
  const nextIndex = step + 1
  const canNext = nextIndex < STEPS.length && (!STEPS[nextIndex].needsResult || hasResult)

  return (
    <section className="mx-auto max-w-6xl px-4 py-10 sm:px-6 sm:py-14">
      <Link to="/modules" className="text-sm text-mist-500 hover:text-mist-300">
        ← All modules
      </Link>
      <div className="mt-5 flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <ModuleGlyph module={module} size="h-14 w-14" />
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-mist-500">
              Module {module.number} · {module.category}
            </p>
            <h1 className="font-display text-3xl font-semibold tracking-tight sm:text-4xl">{module.name}</h1>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {module.method_chips?.map((c) => (
            <Chip key={c}>{c}</Chip>
          ))}
        </div>
      </div>
      <p className="mt-4 max-w-3xl text-mist-300">{module.method}.</p>

      <div ref={topRef} className="mt-8 scroll-mt-20">
        <Stepper step={step} setStep={goto} hasResult={hasResult} />
      </div>

      <div className="mt-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={current.key}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
          >
            {current.key === 'problem' && <ProblemStep module={module} />}
            {current.key === 'why' && <WhyStep module={module} />}
            {current.key === 'math' && <MathStep module={module} />}
            {current.key === 'run' && (
              <RunPanel module={module} run={run} onRun={onRun} onReset={() => setRun(IDLE)} />
            )}
            {current.key === 'replay' && result && <Replay key={result.job_id} replay={result.replay} />}
            {current.key === 'results' && result && <ResultsStep result={result} />}
            {current.key === 'explain' && result && <ExplainStep result={result} />}
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="mt-8 flex items-center justify-between gap-3">
        <Button tone="ghost" disabled={step === 0} onClick={() => goto(step - 1)}>
          ← Back
        </Button>
        {hasResult && result.input && (
          <p className="hidden truncate text-xs text-mist-500 md:block">
            Showing run on <span className="text-mist-300">{result.input.filename}</span>
            {result.input.is_sample ? ' (sample)' : ''}
          </p>
        )}
        {nextIndex < STEPS.length ? (
          <Button disabled={!canNext} onClick={() => goto(nextIndex)} title={!canNext ? 'Run the module first' : undefined}>
            {STEPS[nextIndex].label} →
          </Button>
        ) : (
          <Button tone="ghost" onClick={() => goto(STEPS.findIndex((s) => s.key === 'run'))}>
            Run again
          </Button>
        )}
      </div>
    </section>
  )
}
