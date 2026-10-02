import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ROLE, ROLES, brainReady, dispatch, useCrew } from '../agents/crew'
import { useStore } from '../store'

/**
 * The ops floor's console: who is on the crew, what each of them is doing, and
 * a line to hand any one of them a job.
 *
 * A typed job goes to JARVIS exactly as if you had said it, so the agent you
 * picked starts walking the moment you press enter and the others join in as
 * his tools fire. With the brain offline the agent still runs the job as a
 * demo, which is how the floor gets shown off before anything is connected.
 */
export function OpsPanel() {
  const view = useCrew((s) => s.view)
  const selected = useCrew((s) => s.selected)
  const tasks = useCrew((s) => s.tasks)
  const log = useCrew((s) => s.log)
  const phase = useStore((s) => s.phase)
  const [text, setText] = useState('')
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (selected) input.current?.focus()
  }, [selected])

  if (view !== 'office') {
    return (
      <button className="ops-toggle" onClick={() => useCrew.getState().setView('office')}>
        OPS FLOOR <kbd>O</kbd>
      </button>
    )
  }

  const role = selected ? ROLE[selected] : null
  const live = brainReady()
  const busy = ROLES.filter((r) => tasks[r.id]).length

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    const said = text.trim()
    if (!said || !role) return
    dispatch(role.id, said)
    setText('')
  }

  return (
    <>
      <button className="ops-toggle" onClick={() => useCrew.getState().setView('reactor')}>
        REACTOR <kbd>O</kbd>
      </button>
      <aside className="ops" onPointerDown={(e) => e.stopPropagation()}>
        <div className="ops-title">
          <span>OPS FLOOR</span>
          <span className="ops-count">
            {busy}/{ROLES.length} ACTIVE · {live ? 'LINKED' : phase === 'offline' ? 'DEMO' : 'BOOTING'}
          </span>
        </div>

        <ul className="ops-roster">
          {ROLES.map((r) => {
            const t = tasks[r.id]
            return (
              <li key={r.id}>
                <button
                  className={`ops-agent${selected === r.id ? ' on' : ''}${t ? ' busy' : ''}`}
                  style={{ ['--agent' as string]: r.color }}
                  onClick={() => useCrew.getState().select(selected === r.id ? null : r.id)}
                >
                  <span className="ops-dot" />
                  <span className="ops-name">{r.name}</span>
                  <span className="ops-state">{t ? t.text : r.title}</span>
                </button>
              </li>
            )
          })}
        </ul>

        <AnimatePresence mode="wait">
          {role && (
            <motion.form
              key={role.id}
              className="ops-card"
              style={{ ['--agent' as string]: role.color }}
              onSubmit={submit}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
            >
              <div className="ops-card-head">
                <span className="ops-card-name">{role.name}</span>
                <span className="ops-card-title">{role.title}</span>
              </div>
              <p className="ops-card-blurb">{role.blurb}</p>
              <div className="ops-input">
                <input
                  ref={input}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={`e.g. ${role.example}`}
                  aria-label={`Task for ${role.name}`}
                />
                <button type="submit" disabled={!text.trim()}>
                  ASSIGN
                </button>
              </div>
              <div className="ops-card-foot">
                {tasks[role.id] ? (
                  <button type="button" className="ops-recall" onClick={() => useCrew.getState().finish(role.id)}>
                    RECALL
                  </button>
                ) : (
                  <span />
                )}
                <span className="ops-note">
                  {live ? 'sent to JARVIS' : 'JARVIS offline — demo run'}
                </span>
              </div>
            </motion.form>
          )}
        </AnimatePresence>

        {!role && <div className="ops-hint">click an agent or a name to give them a task · drag to orbit</div>}

        {log.length > 0 && (
          <ol className="ops-log">
            {log.slice(-4).map((e) => (
              <li key={e.id} style={{ ['--agent' as string]: ROLE[e.role].color }}>
                <span className="ops-log-who">{ROLE[e.role].name}</span>
                <span className="ops-log-text">{e.text}</span>
              </li>
            ))}
          </ol>
        )}
      </aside>
    </>
  )
}
