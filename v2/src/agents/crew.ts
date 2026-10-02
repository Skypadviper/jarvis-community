import { create } from 'zustand'
import { useStore, type Phase } from '../store'

/**
 * The crew: JARVIS's work, given bodies.
 *
 * Every tool call the brain makes is routed to one of six agents on the ops
 * floor, who gets up, walks to their desk and works it until the answer is in.
 * The reactor still says *that* he is busy; the floor says *what* he is busy
 * with, and lets you hand a job to a particular agent yourself.
 *
 * This module is plain state. The scene reads it every frame and the overlay
 * subscribes to it; neither owns it, so either view can be swapped out.
 */

export type RoleId = 'research' | 'creative' | 'comms' | 'devices' | 'code' | 'vision'

export type Role = {
  id: RoleId
  name: string
  title: string
  /** Circuitry and nameplate colour. */
  color: string
  blurb: string
  /** One example per role, phrased the way you'd say it. */
  example: string
  /** Matched against the server half of an MCP name, then the tool half. */
  server?: RegExp
  tool?: RegExp
}

/**
 * Order is precedence, the same way src/lib/fillers.ts orders its table: the
 * specific rules first, so `playwright__browser_take_screenshot` is research
 * (a browser) rather than vision (a screenshot).
 */
export const ROLES: Role[] = [
  {
    id: 'research',
    name: 'FRIDAY',
    title: 'Research & web',
    color: '#00e5ff',
    blurb: 'Searches, reads and browses the web.',
    example: 'what happened in AI this week',
    server: /\bexa\b|serper|serpapi|perplexity|tavily|brave|playwright|puppeteer|browserbase|chrome|fetch/,
    tool: /search|\bweb|fetch|crawl|research|browser|navigate/i,
  },
  {
    id: 'creative',
    name: 'KAREN',
    title: 'Images, video & voice',
    color: '#c77dff',
    blurb: 'Generates images, footage and audio.',
    example: 'generate an image of the Mark Seven suit',
    server: /higgsfield|openrouter-image|dalle|flux|midjourney|palmier|heygen|runway|descript|elevenlabs/,
    tool: /image|photo|video|render|upscale|speech|voice|\btts\b|sound/i,
  },
  {
    id: 'comms',
    name: 'EDITH',
    title: 'Mail, calendar & messages',
    color: '#3ef2a8',
    blurb: 'Reads your inbox, diary and messages.',
    example: "summarise what's in my inbox",
    server: /gmail|\bmail\b|calendar|slack|discord|telegram|whatsapp/,
    tool: /mail|inbox|calendar|meeting|message|\bsend\b/i,
  },
  {
    id: 'devices',
    name: 'DUM-E',
    title: 'Phone & home',
    color: '#ffb547',
    blurb: 'Drives your phone, lights and speakers.',
    example: 'take a screenshot of my phone',
    server: /android|\badb\b|simulator|spotify|sonos|^home|homeassistant|\bhue\b/,
    tool: /device|\bphone\b|\bapk\b|lights?|thermostat|\bplay\b|music/i,
  },
  {
    id: 'code',
    name: 'VERONICA',
    title: 'Code & data',
    color: '#ff5d7a',
    blurb: 'Repositories, trackers, files and figures.',
    example: 'open my GitHub notifications',
    server: /github|linear|jira|sentry|mixpanel|posthog|amplitude|clarity|filesystem/,
    tool: /repo|issue|pull_request|commit|query|report|metric|^(bash|read|write|edit|glob|grep)$/i,
  },
  {
    id: 'vision',
    name: 'J.A.R.V.I.S.',
    title: 'Vision & everything else',
    color: '#9fdcff',
    blurb: 'The camera, the screen, and any job nobody else claims.',
    example: 'what am I holding',
    tool: /camera|look|watch|capture|screenshot|vision/i,
  },
]

export const ROLE: Record<RoleId, Role> = Object.fromEntries(
  ROLES.map((r) => [r.id, r]),
) as Record<RoleId, Role>

/** Which agent a tool belongs to. Anything unclaimed goes to the last one. */
export function roleFor(toolName: string): RoleId {
  const parts = toolName.split('__')
  const server = parts.length >= 3 ? parts[1].toLowerCase() : ''
  const tool = (parts.length >= 3 ? parts.slice(2).join('__') : toolName).toLowerCase()
  for (const r of ROLES) {
    if (server && r.server?.test(server)) return r.id
  }
  for (const r of ROLES) {
    if (r.tool?.test(tool)) return r.id
  }
  return 'vision'
}

/** `mcp__exa__web_search_exa` → `web search exa`, for a monitor. */
export function toolLabel(toolName: string): string {
  const parts = toolName.split('__')
  const tool = parts.length >= 3 ? parts.slice(2).join(' ') : toolName
  return tool.replace(/[_-]+/g, ' ').trim()
}

export type Task = {
  id: number
  text: string
  /**
   * 'jarvis' tasks last as long as the turn that started them; 'demo' tasks
   * are what you get with the brain offline, and run on a timer instead.
   */
  source: 'jarvis' | 'demo'
  /**
   * Seconds of desk time a demo task runs for. Counted by the scene while the
   * agent is actually seated, not by the wall clock — on a slow machine the walk
   * alone can take longer than the job, and a timer started at assignment sent
   * agents home before they had ever sat down.
   */
  seconds?: number
}

export type CrewEvent = { id: number; role: RoleId; text: string; at: number }

type Crew = {
  view: 'office' | 'reactor'
  selected: RoleId | null
  tasks: Record<RoleId, Task | null>
  /** Newest last; the overlay shows the tail. */
  log: CrewEvent[]
  /**
   * A command typed into the overlay, waiting for App to hand it to the brain.
   * A counter rather than a bare string so sending the same words twice is
   * still two sends.
   */
  outbox: { n: number; text: string } | null

  setView: (v: Crew['view']) => void
  toggleView: () => void
  select: (r: RoleId | null) => void
  assign: (r: RoleId, text: string, source: Task['source'], ms?: number) => void
  finish: (r: RoleId) => void
  finishSource: (source: Task['source']) => void
  send: (text: string) => void
}

let seq = 0
const none = () =>
  Object.fromEntries(ROLES.map((r) => [r.id, null])) as Record<RoleId, Task | null>

export const useCrew = create<Crew>((set) => ({
  view: 'office',
  selected: null,
  tasks: none(),
  log: [],
  outbox: null,

  setView: (view) => set({ view }),
  toggleView: () => set((s) => ({ view: s.view === 'office' ? 'reactor' : 'office' })),
  select: (selected) => set({ selected }),
  assign: (role, text, source, ms) =>
    set((s) => {
      const task: Task = {
        id: ++seq,
        text,
        source,
        seconds: ms ? ms / 1000 : undefined,
      }
      const log = [...s.log, { id: task.id, role, text, at: Date.now() }].slice(-12)
      return { tasks: { ...s.tasks, [role]: task }, log }
    }),
  finish: (role) => set((s) => ({ tasks: { ...s.tasks, [role]: null } })),
  finishSource: (source) =>
    set((s) => {
      const tasks = { ...s.tasks }
      for (const id of Object.keys(tasks) as RoleId[]) {
        if (tasks[id]?.source === source) tasks[id] = null
      }
      return { tasks }
    }),
  send: (text) => set((s) => ({ outbox: { n: (s.outbox?.n ?? 0) + 1, text } })),
}))

const BUSY: Phase[] = ['thinking', 'tooling', 'speaking']

/** Can a typed command actually reach the brain right now? */
export function brainReady(): boolean {
  const p = useStore.getState().phase
  return p !== 'offline' && p !== 'boot'
}

/**
 * Hand a job to a specific agent. With the brain up, the words go to JARVIS
 * like anything you'd say; with it down the agent still does the walk, so the
 * floor can be shown off before anything is connected.
 */
export function dispatch(role: RoleId, text: string): void {
  const crew = useCrew.getState()
  if (brainReady()) {
    crew.assign(role, text, 'jarvis')
    crew.send(text)
  } else {
    crew.assign(role, text, 'demo', 9000 + Math.random() * 4000)
  }
}

/**
 * Wire the crew to the brain. Called once from App; returns the unsubscribe.
 *
 *   - a tool starting puts its agent to work on it;
 *   - the turn ending (the machine leaving the busy phases) sends every agent
 *     who was working for JARVIS back to the lounge.
 */
export function linkCrew(): () => void {
  return useStore.subscribe((s, prev) => {
    if (s.activeTool && s.activeTool !== prev.activeTool) {
      useCrew.getState().assign(roleFor(s.activeTool), toolLabel(s.activeTool), 'jarvis')
    }
    if (BUSY.includes(prev.phase) && !BUSY.includes(s.phase)) {
      useCrew.getState().finishSource('jarvis')
    }
  })
}
