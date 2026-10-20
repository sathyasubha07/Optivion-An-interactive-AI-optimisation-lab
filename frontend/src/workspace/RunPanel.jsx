import { AnimatePresence, motion } from 'framer-motion'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Button, Chip, ErrorBox, Glass, Spinner } from '../components/ui'

function extOf(name) {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

function Dropzone({ module, file, onFile, disabled }) {
  const input = useRef(null)
  const [over, setOver] = useState(false)
  const preview = useMemo(
    () => (file && file.type.startsWith('image/') && !file.type.includes('tiff') ? URL.createObjectURL(file) : null),
    [file],
  )
  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview)
    },
    [preview],
  )

  const accept = module.accepted_file_types.join(',')
  return (
    <div
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        if (!disabled && e.dataTransfer.files?.[0]) onFile(e.dataTransfer.files[0])
      }}
      className={`flex min-h-48 cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-6 text-center transition ${
        over ? 'border-teal bg-teal/5' : 'border-white/10 hover:border-white/25'
      } ${disabled ? 'pointer-events-none opacity-50' : ''}`}
    >
      <input
        ref={input}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      {preview ? (
        <img src={preview} alt="Selected file preview" className="h-28 w-28 rounded-lg object-cover" />
      ) : (
        <div className="grid h-12 w-12 place-items-center rounded-xl bg-white/5 text-xl text-teal">↑</div>
      )}
      {file ? (
        <div>
          <p className="font-medium text-mist-100">{file.name}</p>
          <p className="font-mono text-xs text-mist-500">{(file.size / 1024).toFixed(1)} KB</p>
        </div>
      ) : (
        <div>
          <p className="font-medium text-mist-100">Drop a file here or click to browse</p>
          <p className="mt-1 text-xs text-mist-500">
            {module.accepted_file_types.join(', ')} · max {module.max_upload_mb} MB
          </p>
        </div>
      )}
    </div>
  )
}

export default function RunPanel({ module, run, onRun, onReset }) {
  const [file, setFile] = useState(null)
  const [clientError, setClientError] = useState(null)
  const busy = run.phase === 'submitting' || run.phase === 'queued' || run.phase === 'running'

  const pickFile = (f) => {
    setClientError(null)
    const ext = extOf(f.name)
    if (!module.accepted_file_types.map((t) => t.toLowerCase()).includes(ext)) {
      setClientError(`Unsupported file type '${ext || '(none)'}'. Accepted: ${module.accepted_file_types.join(', ')}.`)
      return
    }
    if (f.size > module.max_upload_mb * 1024 * 1024) {
      setClientError(`File too large. Maximum is ${module.max_upload_mb} MB.`)
      return
    }
    if (f.size === 0) {
      setClientError('The selected file is empty.')
      return
    }
    setFile(f)
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.3fr_1fr]">
      <Glass className="p-5 sm:p-6">
        <h4 className="font-display font-semibold text-mist-100">Upload your own input</h4>
        <p className="mt-1 text-sm leading-relaxed text-mist-300">{module.input_help}</p>
        <div className="mt-4">
          <Dropzone module={module} file={file} onFile={pickFile} disabled={busy} />
        </div>
        {clientError && <p className="mt-3 text-sm text-rose">{clientError}</p>}
        <div className="mt-4 flex flex-wrap gap-3">
          <Button disabled={!file || busy} onClick={() => onRun({ file })}>
            Run on my file
          </Button>
          {file && !busy && (
            <Button tone="subtle" onClick={() => setFile(null)}>
              Clear
            </Button>
          )}
        </div>
      </Glass>

      <Glass className="p-5 sm:p-6">
        <h4 className="font-display font-semibold text-mist-100">Or try a sample</h4>
        <p className="mt-1 text-sm text-mist-300">Runs the same pipeline on a labelled sample input.</p>
        <div className="mt-4 space-y-2.5">
          {module.samples.length === 0 && <p className="text-sm text-mist-500">No samples are available.</p>}
          {module.samples.map((s) => (
            <button
              key={s.id}
              type="button"
              disabled={busy}
              onClick={() => onRun({ sampleId: s.id })}
              className="glass group block w-full rounded-xl px-4 py-3 text-left transition hover:border-teal/30 disabled:opacity-40"
            >
              <span className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium text-mist-100">{s.label}</span>
                <span className="text-xs font-semibold text-teal opacity-0 transition group-hover:opacity-100">Run →</span>
              </span>
              <span className="mt-0.5 block text-xs leading-snug text-mist-500">{s.description}</span>
            </button>
          ))}
        </div>
      </Glass>

      <AnimatePresence>
        {run.phase !== 'idle' && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="lg:col-span-2"
          >
            {run.phase === 'failed' ? (
              <ErrorBox
                title={run.stage === 'submit' ? 'The run was rejected' : 'The module reported an error'}
                action={
                  <Button tone="ghost" onClick={onReset}>
                    Dismiss
                  </Button>
                }
              >
                {run.error}
              </ErrorBox>
            ) : (
              <Glass className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3">
                  {busy ? <Spinner className="h-5 w-5 text-teal" /> : <span className="text-teal">✓</span>}
                  <div>
                    <p className="text-sm font-medium text-mist-100">
                      {run.phase === 'submitting' && 'Uploading…'}
                      {run.phase === 'queued' && 'Queued behind another run…'}
                      {run.phase === 'running' && 'Solver running on the backend…'}
                      {run.phase === 'succeeded' && 'Run complete. Results are ready.'}
                    </p>
                    <p className="text-xs text-mist-500">
                      Input: {run.inputLabel}
                      {typeof run.elapsed === 'number' && ` · ${run.elapsed.toFixed(1)} s`}
                    </p>
                  </div>
                </div>
                <Chip tone={busy ? 'violet' : 'teal'}>{run.phase}</Chip>
              </Glass>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
