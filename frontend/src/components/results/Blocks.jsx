import { useState } from 'react'
import { Bar, BarChart, Cell, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { formatValue } from '../../lib/format'
import { Glass } from '../ui'

export function BarChartCard({ chart }) {
  const data = (chart.data ?? []).filter((d) => typeof d.value === 'number')
  if (!data.length) return null
  const height = Math.max(220, data.length * 22 + 40)
  return (
    <Glass className="p-5">
      <h4 className="font-display font-semibold text-mist-100">{chart.title}</h4>
      {chart.note && <p className="mt-1 text-xs leading-relaxed text-mist-500">{chart.note}</p>}
      <div className="mt-4" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" horizontal={false} />
            <XAxis type="number" stroke="#7d88a6" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v) => formatValue(v, 'sci')} />
            <YAxis
              type="category"
              dataKey="label"
              stroke="#7d88a6"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              width={150}
              tickFormatter={(s) => (s.length > 22 ? `${s.slice(0, 21)}…` : s)}
            />
            <ReferenceLine x={0} stroke="rgba(255,255,255,0.25)" />
            <Tooltip
              cursor={{ fill: 'rgba(255,255,255,0.04)' }}
              contentStyle={{ background: '#0a0e19', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, fontSize: 12 }}
              itemStyle={{ color: '#eef2fb' }}
              labelStyle={{ color: '#b8c1d9' }}
              formatter={(v) => [formatValue(v, 'float4'), chart.y_label || 'value']}
            />
            <Bar dataKey="value" radius={[4, 4, 4, 4]} isAnimationActive>
              {data.map((d) => (
                <Cell key={d.label} fill={d.value >= 0 ? '#fb7185' : '#5eead4'} fillOpacity={d.value === 0 ? 0.25 : 0.85} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Glass>
  )
}

function Cell_({ value, fmt }) {
  if (value === null || value === undefined || value === '') return <span className="text-mist-500">—</span>
  if (fmt === 'bool' && typeof value === 'boolean')
    return <span className={value ? 'text-teal' : 'text-amber'}>{value ? 'Yes' : 'No'}</span>
  if (fmt) return <span className="font-mono">{formatValue(value, fmt)}</span>
  return <span>{String(value)}</span>
}

export function DataTable({ table }) {
  const [expanded, setExpanded] = useState(false)
  const limit = 12
  const rows = table.rows ?? []
  const visible = expanded ? rows : rows.slice(0, limit)
  return (
    <Glass className="overflow-hidden">
      <div className="px-5 pt-5">
        <h4 className="font-display font-semibold text-mist-100">{table.title}</h4>
        {table.note && <p className="mt-1 text-xs leading-relaxed text-mist-500">{table.note}</p>}
      </div>
      <div className="scroll-thin mt-4 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-wider text-mist-500">
            <tr>
              {table.columns.map((c) => (
                <th key={c.key} className="whitespace-nowrap px-5 py-2 font-medium">
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((row, i) => (
              <tr key={i} className="border-t border-white/5">
                {table.columns.map((c) => (
                  <td key={c.key} className="px-5 py-2.5 align-top text-mist-300">
                    <Cell_ value={row[c.key]} fmt={c.fmt} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > limit && (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="w-full border-t border-white/5 py-3 text-xs font-semibold text-teal hover:bg-white/5"
        >
          {expanded ? 'Show fewer rows' : `Show all ${rows.length} rows`}
        </button>
      )}
    </Glass>
  )
}

export function Notes({ notes = [] }) {
  if (!notes.length) return null
  return (
    <Glass className="p-5">
      <h4 className="font-display font-semibold text-mist-100">Provenance and honesty notes</h4>
      <ul className="mt-3 space-y-2 text-sm leading-relaxed text-mist-300">
        {notes.map((n, i) => (
          <li key={i} className="flex gap-2">
            <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-teal" />
            <span>{n}</span>
          </li>
        ))}
      </ul>
    </Glass>
  )
}

export function Downloads({ downloads = [] }) {
  if (!downloads.length) return <p className="text-sm text-mist-500">This run produced no downloadable files.</p>
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {downloads.map((d) => (
        <a
          key={d.url}
          href={d.url}
          download={d.filename}
          className="glass group flex items-center justify-between gap-3 rounded-xl px-4 py-3 text-sm transition hover:border-teal/30"
        >
          <span>
            <span className="block font-medium text-mist-100">{d.label}</span>
            <span className="block font-mono text-xs text-mist-500">{d.filename}</span>
          </span>
          <span className="text-teal transition group-hover:translate-y-0.5" aria-hidden="true">
            ↓
          </span>
        </a>
      ))}
    </div>
  )
}
