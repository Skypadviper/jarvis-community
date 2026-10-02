import { Suspense, useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber'
import { Html, OrbitControls, useGLTF } from '@react-three/drei'
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { clone as cloneSkinned } from 'three/examples/jsm/utils/SkeletonUtils.js'
import { useStore, accentFor } from '../store'
import { ROLES, useCrew, type Role, type RoleId } from '../agents/crew'

/**
 * The ops floor: JARVIS's crew at their desks.
 *
 * The models come out of blender/build_agents.py — one rigged agent with Idle,
 * Walk, Sit, Type and Wave clips, and one workstation whose chair sits at
 * z = -0.6 with an agent in it facing +Z. Everything about *where* things go is
 * here; everything about what they look like is in that script.
 *
 * Like the reactor scene, nothing that changes per frame goes through React.
 * Each agent is a small state machine stepped inside useFrame, reading the crew
 * store imperatively; React only mounts things once.
 */

const AGENT_URL = '/models/agent.glb'
const STATION_URL = '/models/workstation.glb'

/** Desks on an arc behind the hub, screens facing the camera. */
const DESK_R = 5
/** The walkway between the lounge and the desks. */
const AISLE_R = 3.5
/** Where the crew stands about when there is nothing to do. */
const LOUNGE_R = 2.7
const DESK_ANGLES = [-75, -45, -15, 15, 45, 75]
const WALK_SPEED = 1.05
const SIT_HOLD = 0.9
const WAVE_HOLD = 1.6

const rad = THREE.MathUtils.degToRad
/** Angle 0 is straight back (-Z); 180 is toward the camera. */
const onRing = (r: number, deg: number) =>
  new THREE.Vector3(Math.sin(rad(deg)) * r, 0, -Math.cos(rad(deg)) * r)

type Station = {
  pos: THREE.Vector3
  yaw: number
  chair: THREE.Vector3
  /** Beside the chair, so nobody walks through its back to sit down. */
  beside: THREE.Vector3
}

function layout(i: number): Station {
  const a = DESK_ANGLES[i]
  const pos = onRing(DESK_R, a)
  const out = pos.clone().normalize()
  const yaw = Math.atan2(out.x, out.z)
  const chair = pos.clone().addScaledVector(out, -0.6)
  const side = new THREE.Vector3(out.z, 0, -out.x) // local +X
  const beside = chair.clone().addScaledVector(side, 0.65).addScaledVector(out, -0.1)
  return { pos, yaw, chair, beside }
}

function loungeAngle(i: number) {
  return 180 - DESK_ANGLES[i] * (2 / 3)
}

/** Lounge → aisle → round the arc → beside the chair → into it. */
function routeToDesk(i: number, from: THREE.Vector3): THREE.Vector3[] {
  const st = layout(i)
  const start = Math.atan2(from.x, -from.z) * (180 / Math.PI)
  let a0 = ((start % 360) + 360) % 360
  let a1 = ((DESK_ANGLES[i] % 360) + 360) % 360
  // Always go round the side, never across the front of the hub.
  if (DESK_ANGLES[i] > 0 && a1 > a0) a1 -= 360
  if (DESK_ANGLES[i] < 0 && a1 < a0) a0 -= 360
  const pts = [onRing(AISLE_R, a0)]
  const steps = Math.max(1, Math.round(Math.abs(a1 - a0) / 12))
  for (let k = 1; k <= steps; k++) pts.push(onRing(AISLE_R, a0 + ((a1 - a0) * k) / steps))
  pts.push(st.beside, st.chair)
  return pts
}

function routeToLounge(i: number): THREE.Vector3[] {
  const st = layout(i)
  return [st.beside, ...routeToDesk(i, onRing(LOUNGE_R, loungeAngle(i))).slice(0, -2).reverse(), onRing(LOUNGE_R, loungeAngle(i))]
}

type Mode = 'lounge' | 'toDesk' | 'sitting' | 'working' | 'standing' | 'toLounge'

type Agent = {
  role: Role
  index: number
  root: THREE.Group
  mixer: THREE.AnimationMixer
  actions: Record<string, THREE.AnimationAction>
  current: string
  mode: Mode
  path: THREE.Vector3[]
  timer: number
  taskId: number | null
  /** Seconds spent seated on the current task. */
  worked: number
  yaw: number
  screen: Screen
  glow: THREE.MeshStandardMaterial[]
}

function play(a: Agent, clip: string, fade = 0.35) {
  if (a.current === clip) return
  const next = a.actions[clip]
  const prev = a.actions[a.current]
  next.reset().setEffectiveWeight(1).fadeIn(fade).play()
  prev?.fadeOut(fade)
  a.current = clip
}

/** Shortest-way-round yaw easing. */
function easeYaw(from: number, to: number, k: number) {
  let d = to - from
  d = Math.atan2(Math.sin(d), Math.cos(d))
  return from + d * k
}

/**
 * Walk the path; true when there is no path left. Turns before it moves so a
 * sharp corner reads as a pivot rather than a moonwalk.
 */
function walk(a: Agent, dt: number): boolean {
  const target = a.path[0]
  if (!target) return true
  const p = a.root.position
  const d = new THREE.Vector3().subVectors(target, p)
  d.y = 0
  const dist = d.length()
  if (dist < 0.06) {
    a.path.shift()
    return a.path.length === 0
  }
  const want = Math.atan2(d.x, d.z)
  a.yaw = easeYaw(a.yaw, want, Math.min(1, dt * 7))
  const facing = Math.cos(want - a.yaw)
  p.addScaledVector(d.normalize(), Math.min(dist, WALK_SPEED * dt * Math.max(0.2, facing)))
  return false
}

// -- monitor ------------------------------------------------------------------

type Screen = {
  mesh: THREE.Mesh
  tex: THREE.CanvasTexture
  ctx: CanvasRenderingContext2D
  key: string
}

function makeScreen(): Screen {
  const canvas = document.createElement('canvas')
  canvas.width = 512
  canvas.height = 288
  const ctx = canvas.getContext('2d')!
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(0.76, 0.42),
    new THREE.MeshBasicMaterial({ map: tex, toneMapped: false }),
  )
  // The GLB's own screen faces -Z at y 1.07; sit just in front of it.
  mesh.position.set(0, 1.07, 0.166)
  mesh.rotation.y = Math.PI
  return { mesh, tex, ctx, key: '' }
}

function wrap(ctx: CanvasRenderingContext2D, text: string, width: number): string[] {
  const words = text.split(/\s+/)
  const lines: string[] = []
  let line = ''
  for (const w of words) {
    const next = line ? `${line} ${w}` : w
    if (ctx.measureText(next).width > width && line) {
      lines.push(line)
      line = w
    } else line = next
  }
  if (line) lines.push(line)
  return lines.slice(0, 4)
}

/** Redrawn only when what it says changes, plus a cheap scan line per frame. */
function drawScreen(s: Screen, role: Role, task: string | null, t: number) {
  const { ctx } = s
  const key = `${task ?? ''}|${Math.floor(t * 8)}`
  if (key === s.key) return
  s.key = key
  const W = 512
  const H = 288
  ctx.fillStyle = '#010b12'
  ctx.fillRect(0, 0, W, H)
  ctx.strokeStyle = role.color
  ctx.globalAlpha = 0.25
  for (let y = 0; y < H; y += 6) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(W, y)
    ctx.stroke()
  }
  ctx.globalAlpha = 1
  ctx.fillStyle = role.color
  ctx.font = '600 30px "Chakra Petch", sans-serif'
  ctx.fillText(role.name, 24, 46)
  ctx.font = '300 18px "JetBrains Mono", monospace'
  ctx.globalAlpha = 0.7
  ctx.fillText(role.title.toUpperCase(), 24, 74)
  ctx.globalAlpha = 1
  if (task) {
    ctx.font = '400 24px "Chakra Petch", sans-serif'
    ctx.fillStyle = '#e6fdff'
    wrap(ctx, task, W - 48).forEach((l, i) => ctx.fillText(l, 24, 128 + i * 32))
    // A progress sweep, so a long job never looks frozen.
    const x = ((t * 0.35) % 1) * (W - 48)
    ctx.fillStyle = role.color
    ctx.fillRect(24, H - 34, W - 48, 3)
    ctx.globalAlpha = 0.9
    ctx.fillRect(24 + x, H - 40, 60, 15)
    ctx.globalAlpha = 1
  } else {
    ctx.font = '300 20px "JetBrains Mono", monospace'
    ctx.globalAlpha = 0.5 + 0.2 * Math.sin(t * 2)
    ctx.fillText('STANDING BY', 24, 150)
    ctx.globalAlpha = 1
  }
  s.tex.needsUpdate = true
}

// -- the floor ----------------------------------------------------------------

function Crew() {
  const agentGltf = useGLTF(AGENT_URL)
  const stationGltf = useGLTF(STATION_URL)
  const group = useRef<THREE.Group>(null)

  const agents = useMemo<Agent[]>(() => {
    return ROLES.map((role, index) => {
      const root = new THREE.Group()
      const body = cloneSkinned(agentGltf.scene) as THREE.Group
      const glow: THREE.MeshStandardMaterial[] = []
      const tint = new THREE.Color(role.color)
      body.traverse((o) => {
        const m = o as THREE.Mesh
        if (!m.isMesh) return
        m.castShadow = true
        m.userData.role = role.id
        const mats = (Array.isArray(m.material) ? m.material : [m.material]).map((mat) => {
          const c = (mat as THREE.MeshStandardMaterial).clone()
          if (c.name === 'Glow' || c.name === 'Visor') {
            c.emissive.copy(tint)
            c.color.copy(tint).multiplyScalar(0.4)
            glow.push(c)
          }
          return c
        })
        m.material = Array.isArray(m.material) ? mats : mats[0]
      })
      root.add(body)
      const mixer = new THREE.AnimationMixer(body)
      const actions: Record<string, THREE.AnimationAction> = {}
      for (const clip of agentGltf.animations) actions[clip.name] = mixer.clipAction(clip)
      actions.Walk?.setEffectiveTimeScale(1.35)
      const start = onRing(LOUNGE_R, loungeAngle(index))
      root.position.copy(start)
      const yaw = Math.atan2(-start.x * 0.3, 1)
      const a: Agent = {
        role,
        index,
        root,
        mixer,
        actions,
        current: 'Idle',
        mode: 'lounge',
        path: [],
        timer: 0,
        taskId: null,
        worked: 0,
        yaw,
        screen: makeScreen(),
        glow,
      }
      actions.Idle.play()
      // Out of step with each other, or six agents breathe in unison.
      mixer.update(Math.random() * 2)
      return a
    })
  }, [agentGltf])

  const stations = useMemo(
    () =>
      ROLES.map((role, i) => {
        const st = layout(i)
        const g = stationGltf.scene.clone(true)
        g.position.copy(st.pos)
        g.rotation.y = st.yaw
        g.traverse((o) => {
          const m = o as THREE.Mesh
          if (!m.isMesh) return
          m.receiveShadow = true
          m.castShadow = true
        })
        const a = agents[i]
        g.add(a.screen.mesh)
        drawScreen(a.screen, role, null, 0)
        return g
      }),
    [stationGltf, agents],
  )

  useEffect(
    () => () => {
      for (const a of agents) {
        a.mixer.stopAllAction()
        a.screen.tex.dispose()
      }
    },
    [agents],
  )

  useFrame((state, dt) => {
    dt = Math.min(dt, 0.1)
    const t = state.clock.elapsedTime
    const crew = useCrew.getState()

    for (const a of agents) {
      const task = crew.tasks[a.role.id]
      if (task?.seconds && a.worked > task.seconds) crew.finish(a.role.id)
      const want = task ? task.id : null

      // React to the job changing, whatever we were doing at the time.
      if (want !== a.taskId) {
        const had = a.taskId
        a.taskId = want
        a.worked = 0
        if (want !== null && had === null) {
          if (a.mode === 'lounge' || a.mode === 'toLounge') {
            a.mode = 'toDesk'
            a.path = routeToDesk(a.index, a.root.position)
            play(a, 'Walk')
          } else if (a.mode === 'standing') {
            // Called back before they'd left the chair.
            a.mode = 'sitting'
            a.timer = 0
            play(a, 'Sit', 0.4)
          }
          // Already at (or heading to) the desk: just keep going.
        } else if (want === null) {
          if (a.mode === 'working' || a.mode === 'sitting') {
            a.mode = 'standing'
            a.timer = 0
            play(a, 'Idle', 0.5)
          } else if (a.mode === 'toDesk') {
            a.mode = 'toLounge'
            a.path = routeToLounge(a.index)
          }
        }
      }

      const st = layout(a.index)
      switch (a.mode) {
        case 'lounge': {
          // Turn to face whoever is selected, otherwise the camera.
          const sel = crew.selected
          let face = 0
          if (sel && sel !== a.role.id) {
            const other = agents.find((o) => o.role.id === sel)!
            face = Math.atan2(other.root.position.x - a.root.position.x, other.root.position.z - a.root.position.z)
          }
          a.yaw = easeYaw(a.yaw, face, Math.min(1, dt * 2))
          break
        }
        case 'toDesk':
        case 'toLounge':
          if (walk(a, dt)) {
            if (a.mode === 'toDesk') {
              a.mode = 'sitting'
              a.timer = 0
              play(a, 'Sit', 0.4)
            } else {
              a.mode = 'lounge'
              play(a, 'Idle', 0.4)
            }
          }
          break
        case 'sitting':
          a.root.position.lerp(st.chair, Math.min(1, dt * 6))
          a.yaw = easeYaw(a.yaw, st.yaw, Math.min(1, dt * 8))
          a.timer += dt
          if (a.timer > SIT_HOLD) {
            a.mode = 'working'
            play(a, 'Type', 0.5)
          }
          break
        case 'working':
          a.worked += dt
          a.root.position.lerp(st.chair, Math.min(1, dt * 6))
          a.yaw = easeYaw(a.yaw, st.yaw, Math.min(1, dt * 8))
          break
        case 'standing':
          // Up, a wave to say it's done, then back to the lounge.
          a.timer += dt
          if (a.timer > 0.5 && a.current !== 'Wave') {
            a.yaw = easeYaw(a.yaw, 0, 1)
            play(a, 'Wave', 0.4)
          }
          if (a.timer > 0.5 + WAVE_HOLD) {
            a.mode = 'toLounge'
            a.path = routeToLounge(a.index)
            play(a, 'Walk')
          }
          break
      }
      a.root.rotation.y = a.yaw
      a.mixer.update(dt)

      // Brighter while working; selected agents pulse.
      const busy = a.mode === 'working' || a.mode === 'sitting'
      const sel = crew.selected === a.role.id
      const glow = (busy ? 7 : 3.5) + (sel ? 2.5 * (0.5 + 0.5 * Math.sin(t * 5)) : 0)
      for (const m of a.glow) m.emissiveIntensity += (glow - m.emissiveIntensity) * Math.min(1, dt * 4)

      drawScreen(a.screen, a.role, busy || a.mode === 'toDesk' ? (task?.text ?? null) : null, t)
    }
  })

  const pick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation()
    let o: THREE.Object3D | null = e.object
    while (o && !o.userData.role) o = o.parent
    const id = o?.userData.role as RoleId | undefined
    if (id) useCrew.getState().select(useCrew.getState().selected === id ? null : id)
  }

  return (
    <group ref={group}>
      {stations.map((g, i) => (
        <primitive key={`s${i}`} object={g} />
      ))}
      {agents.map((a) => (
        <group key={a.role.id}>
          <primitive
            object={a.root}
            onClick={pick}
            onPointerOver={(e: ThreeEvent<PointerEvent>) => {
              e.stopPropagation()
              document.body.style.cursor = 'pointer'
            }}
            onPointerOut={() => (document.body.style.cursor = '')}
          />
          <Nameplate agent={a} />
        </group>
      ))}
    </group>
  )
}

/** A label that floats over an agent's head and follows them about. */
function Nameplate({ agent }: { agent: Agent }) {
  const ref = useRef<THREE.Group>(null)
  const selected = useCrew((s) => s.selected === agent.role.id)
  const task = useCrew((s) => s.tasks[agent.role.id])
  useFrame(() => {
    if (!ref.current) return
    ref.current.position.set(agent.root.position.x, 2.05, agent.root.position.z)
  })
  return (
    <group ref={ref}>
      <Html center distanceFactor={9} zIndexRange={[20, 0]}>
        <button
          className={`agent-tag${selected ? ' on' : ''}${task ? ' busy' : ''}`}
          style={{ ['--agent' as string]: agent.role.color }}
          onClick={() => useCrew.getState().select(selected ? null : agent.role.id)}
        >
          <span className="agent-tag-name">{agent.role.name}</span>
          <span className="agent-tag-state">{task ? 'working' : 'ready'}</span>
        </button>
      </Html>
    </group>
  )
}

/** JARVIS himself: the hub at the centre of the floor, in the phase colour. */
function Hub() {
  const ring = useRef<THREE.Mesh>(null)
  const ring2 = useRef<THREE.Mesh>(null)
  const core = useRef<THREE.Mesh>(null)
  const light = useRef<THREE.PointLight>(null)
  const color = useMemo(() => new THREE.Color('#12908f'), [])
  const target = useMemo(() => new THREE.Color(), [])
  const mats = useMemo(
    () => ({
      ring: new THREE.MeshBasicMaterial({ color: '#00e5ff', toneMapped: false, transparent: true, opacity: 0.9 }),
      core: new THREE.MeshBasicMaterial({ color: '#00e5ff', toneMapped: false, transparent: true, opacity: 0.55 }),
      base: new THREE.MeshStandardMaterial({ color: '#06121c', metalness: 0.8, roughness: 0.3 }),
    }),
    [],
  )
  useFrame((state, dt) => {
    const { phase, level, ui } = useStore.getState()
    target.set(accentFor(phase, ui))
    color.lerp(target, Math.min(1, dt * 3))
    mats.ring.color.copy(color)
    mats.core.color.copy(color)
    const t = state.clock.elapsedTime
    const busy = phase === 'thinking' || phase === 'tooling' || phase === 'speaking'
    if (ring.current) ring.current.rotation.z += dt * (busy ? 2.4 : 0.5)
    if (ring2.current) ring2.current.rotation.z -= dt * (busy ? 1.6 : 0.3)
    if (core.current) core.current.scale.setScalar(0.32 + level * 0.18 + 0.02 * Math.sin(t * 2))
    if (light.current) {
      light.current.color.copy(color)
      light.current.intensity = 6 + level * 10 + (busy ? 4 : 0)
    }
  })
  return (
    <group>
      <mesh material={mats.base} position={[0, 0.15, 0]} receiveShadow>
        <cylinderGeometry args={[0.9, 1.05, 0.3, 48]} />
      </mesh>
      <group position={[0, 1.55, 0]}>
        <mesh ref={core} material={mats.core}>
          <icosahedronGeometry args={[1, 3]} />
        </mesh>
        <mesh ref={ring} material={mats.ring} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[0.62, 0.012, 8, 96]} />
        </mesh>
        <mesh ref={ring2} material={mats.ring} rotation={[Math.PI / 2.4, 0.3, 0]}>
          <torusGeometry args={[0.78, 0.008, 8, 96]} />
        </mesh>
      </group>
      <pointLight ref={light} position={[0, 1.6, 0]} distance={9} decay={1.6} />
    </group>
  )
}

function Floor() {
  const grid = useMemo(() => {
    const g = new THREE.GridHelper(30, 60, '#0b5560', '#06303a')
    const m = g.material as THREE.LineBasicMaterial
    m.transparent = true
    m.opacity = 0.55
    g.position.y = 0.002
    return g
  }, [])
  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow onClick={() => useCrew.getState().select(null)}>
        <circleGeometry args={[15, 64]} />
        <meshStandardMaterial color="#02080e" metalness={0.6} roughness={0.45} />
      </mesh>
      <primitive object={grid} />
      {/* Walkway and lounge rings, painted on the floor. */}
      {[AISLE_R, LOUNGE_R].map((r) => (
        <mesh key={r} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.004, 0]}>
          <ringGeometry args={[r - 0.015, r + 0.015, 128]} />
          <meshBasicMaterial color="#00e5ff" transparent opacity={0.25} toneMapped={false} />
        </mesh>
      ))}
    </group>
  )
}

export function Office() {
  return (
    <Canvas
      className="scene office"
      shadows
      dpr={[1, 2]}
      camera={{ position: [0, 5.2, 8.6], fov: 42 }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={['#01060c']} />
      <fog attach="fog" args={['#01060c', 12, 26]} />
      <hemisphereLight args={['#9fdcff', '#02080e', 0.9]} />
      <directionalLight
        position={[4, 9, 6]}
        intensity={1.6}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-8}
        shadow-camera-right={8}
        shadow-camera-top={8}
        shadow-camera-bottom={-8}
      />
      <Floor />
      <Hub />
      <Suspense fallback={null}>
        <Crew />
      </Suspense>
      <OrbitControls
        makeDefault
        target={[0, 0.9, -1.2]}
        enablePan={false}
        minDistance={3}
        maxDistance={16}
        maxPolarAngle={Math.PI / 2.15}
      />
      <EffectComposer multisampling={0}>
        <Bloom intensity={0.9} luminanceThreshold={0.55} luminanceSmoothing={0.4} mipmapBlur radius={0.6} />
        <Vignette eskil={false} offset={0.25} darkness={0.85} />
      </EffectComposer>
    </Canvas>
  )
}

useGLTF.preload(AGENT_URL)
useGLTF.preload(STATION_URL)
