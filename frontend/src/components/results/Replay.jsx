import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatValue, seriesIsUsable } from '../../lib/format'
import { Button, Chip, Glass } from '../ui'

const COLORS = ['#5eead4', '#a78bfa', '#fbbf24', '#93c5fd', '#fb7185', '#86efac']
const PLAY_MS = 6000 // a full replay takes ~6 s regardless of iteration count

const axisProps = { stroke: '#7d88a6', fontSize: 11, tickLine: false, axisLine: false }

function SeriesChart({ series, color, step, xLabel }) {
  const numeric = series.values.filter((v) => typeof v === 'number')
  const log = series.log_scale && numeric.length > 0 && numeric.every((v) => v > 0)
  const data = useMemo(
    () => series.values.map((v, i) => ({ k: i + 1, all: v, shown: i < step ? v : null })),
    [series.values, step],
  )
  const current = step > 0 ? series.values[step - 1] : null

  return (
    <Glass className="p-4">
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm font-medium text-mist-100">{series.label}</p>
        <p className="font-mono text-sm" style={{ color }}>
          {step > 0 ? formatValue(current, 'sci') : '—'}
        </p>
      </div>
      <div className="mt-3 h-40">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis dataKey="k" {...axisProps} type="number" domain={[1, Math.max(1, series.values.length)]} allowDecimals={false} />
            <YAxis
              {...axisProps}
              width={58}
              scale={log ? 'log' : 'auto'}
              domain={log ? ['auto', 'auto'] : ['auto', 'auto']}
              tickFormatter={(v) => formatValue(v, 'sci')}
              allowDataOverflow={log}
            />
            <Tooltip
              contentStyle={{ background: '#0a0e19', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 12 }}
              itemStyle={{ color: '#eef2fb' }}
              labelStyle={{ color: '#b8c1d9' }}
              labelFormatter={(k) => `${xLabel} ${k}`}
              formatter={(v, name) => (name === 'shown' ? [formatValue(v, 'sci'), series.label] : [null, null])}
            />
            <Line dataKey="all" stroke={color} strokeOpacity={0.18} dot={false} strokeWidth={1.5} isAnimationActive={false} />
            <Line dataKey="shown" stroke={color} dot={false} strokeWidth={2.2} isAnimationActive={false} connectNulls={false} />
            {step > 0 && typeof current === 'number' && (
              <ReferenceDot x={step} y={current} r={4} fill={color} stroke="#06080f" strokeWidth={2} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {log && <p className="mt-1 text-[11px] text-mist-500">log scale</p>}
    </Glass>
  )
}

export default function Replay({ replay, finalIterations }) {
  const series = (replay?.series ?? []).filter((s) => seriesIsUsable(s.values))
  const n = series.reduce((m, s) => Math.max(m, s.values.length), 0)
  const [step, setStep] = useState(n)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    if (!playing) return undefined
    const interval = Math.max(25, PLAY_MS / Math.max(1, n))
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= n) {
          setPlaying(false)
          return n
        }
        return s + 1
      })
    }, interval)
    return () => clearInterval(id)
  }, [playing, n])

  if (!replay?.available || series.length === 0) {
    return (
      <Glass className="p-6 sm:p-8">
        <div className="flex flex-wrap items-center gap-2">
          <Chip tone="amber">No recorded iterations</Chip>
          {finalIterations !== undefined && <Chip>Final iteration count: {formatValue(finalIterations, 'int')}</Chip>}
        </div>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-mist-300">
          {replay?.note || 'This module did not record per-iteration values, so there is nothing to replay.'}
        </p>
      </Glass>
    )
  }

  const play = () => {
    if (step >= n) setStep(0)
    setPlaying(true)
  }

  return (
    <div className="space-y-5">
      <Glass className="p-5">
        <div className="flex flex-wrap items-center gap-2">
          <Chip tone="violet">● Replay of recorded iterations</Chip>
          <span className="text-xs text-mist-500">{replay.note}</span>
        </div>
        <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center">
          <div className="flex gap-2">
            {playing ? (
              <Button tone="ghost" onClick={() => setPlaying(false)} aria-label="Pause replay">
                ❚❚ Pause
              </Button>
            ) : (
              <Button onClick={play} aria-label="Play replay">
                ▶ {step >= n ? 'Replay' : 'Play'}
              </Button>
            )}
            <Button
              tone="ghost"
              onClick={() => {
                setPlaying(false)
                setStep(0)
              }}
              aria-label="Reset replay"
            >
              ↺
            </Button>
          </div>
          <input
            type="range"
            min="0"
            max={n}
            value={step}
            onChange={(e) => {
              setPlaying(false)
              setStep(Number(e.target.value))
            }}
            className="scrubber flex-1"
            aria-label="Iteration"
          />
          <p className="shrink-0 font-mono text-sm text-mist-300">
            {replay.x_label} <span className="text-mist-100">{step}</span> / {n}
          </p>
        </div>
      </Glass>
      <div className="grid gap-4 md:grid-cols-2">
        {series.map((s, i) => (
          <SeriesChart key={s.key} series={s} color={COLORS[i % COLORS.length]} step={step} xLabel={replay.x_label} />
        ))}
      </div>
    </div>
  )
}
