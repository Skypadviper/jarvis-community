import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { useStore, accentFor } from '../store'

/**
 * JARVIS's brain: 70,000 glowing particles scattered over the surface sculpted
 * by blender/build_brain.py, standing on the hub in the middle of the ops floor.
 *
 * Each particle knows its surface normal and whether it sits on a ridge (gyrus)
 * or down a groove (sulcus), so the folds come through as light and shadow even
 * though there is no mesh at all. Particles on the far side fade out, which is
 * what keeps it reading as a sculpture rather than a cloud.
 *
 * Everything moves in the vertex shader, which is what lets the whole brain
 * burst apart when a job finishes and rebuild itself a moment later.
 */

const URL = `${import.meta.env.BASE_URL}models/brain-points.bin`
const STRIDE = 8 // x, y, z, nx, ny, nz, fold, region

/** Burst timeline, in seconds: fly out, hang, rebuild. */
const OUT = 0.6
const HOLD_END = 1.4
const BACK_END = 2.9

/** World-space diameter of one particle. */
const SIZE = 0.0115

/**
 * Burst requests, from anywhere. Module state rather than React state because
 * the scene reads it every frame and nothing needs to re-render when it changes.
 * Bursts that arrive mid-burst wait their turn, so the brain never snaps from
 * half-rebuilt straight back to exploding.
 */
const fx = { t: 99, color: new THREE.Color('#00e5ff'), queue: [] as string[] }

/**
 * How burst the brain is right now, 0 (whole) to 1 (fully exploded), and in
 * whose colour, for anything that wants to react with it, like the capsule.
 */
export function burstState(): { k: number; color: THREE.Color } {
  const t = fx.t
  const fly = 1 - Math.pow(1 - Math.min(1, Math.max(0, t / OUT)), 3)
  const x = Math.min(1, Math.max(0, (t - HOLD_END) / (BACK_END - HOLD_END)))
  const back = x * x * (3 - 2 * x)
  return { k: fly * (1 - back), color: fx.color }
}

export function burstBrain(color = '#00e5ff') {
  if (fx.t > BACK_END) {
    fx.t = 0
    fx.color.set(color)
  } else if (fx.queue.length < 3) {
    fx.queue.push(color)
  }
}

const vertex = /* glsl */ `
  attribute vec3 aNormal;
  attribute float aFold;
  attribute float aRegion;
  attribute float aSeed;

  uniform float uTime;
  uniform float uBurst;
  uniform float uLevel;
  uniform float uBusy;
  uniform float uViewH;
  uniform vec3 uColor;
  uniform vec3 uBurstColor;

  varying vec3 vColor;
  varying float vAlpha;

  vec3 hash3(float n) {
    return fract(sin(vec3(n, n + 1.7, n + 3.1)) * vec3(43758.5453, 22578.1459, 19642.3490));
  }

  void main() {
    vec3 rnd = hash3(aSeed);

    // -- burst: ease out, hang, ease back --------------------------------
    float t = uBurst;
    float fly = 1.0 - pow(1.0 - clamp(t / ${OUT.toFixed(2)}, 0.0, 1.0), 3.0);
    float back = smoothstep(${HOLD_END.toFixed(2)}, ${BACK_END.toFixed(2)}, t);
    float k = fly * (1.0 - back);

    vec3 dir = normalize(aNormal * 0.8 + normalize(position) * 0.6 + (rnd - 0.5) * 0.9);
    float dist = 0.35 + rnd.x * rnd.x * 2.2;
    vec3 drift = vec3(0.0, -0.2, 0.0) * k * clamp(t, 0.0, ${HOLD_END.toFixed(2)});
    // A faint shimmer at rest, so the surface is never perfectly still.
    vec3 shimmer = aNormal * 0.003 * sin(uTime * 2.3 + aSeed * 0.37);
    vec3 world = (position + shimmer + dir * dist * k + drift) * (1.0 + 0.03 * uLevel);

    vec4 mv = modelViewMatrix * vec4(world, 1.0);
    gl_Position = projectionMatrix * mv;
    // Constant world size, whatever the camera distance and screen height.
    gl_PointSize = ${SIZE.toFixed(4)} * projectionMatrix[1][1] * uViewH * 0.5 / -mv.z * (1.0 + 0.9 * k);

    // -- light ------------------------------------------------------------
    vec3 n = normalize(normalMatrix * aNormal);
    float facing = n.z;
    float lambert = max(dot(n, normalize(vec3(-0.35, 0.75, 0.55))), 0.0);
    float rim = pow(1.0 - clamp(facing, 0.0, 1.0), 3.0);
    // Grooves darker than ridges, so the folds read as carved.
    float fold = smoothstep(0.1, 0.95, aFold);

    // Activity: waves running across the lobes, quicker while working.
    float speed = 1.4 + 4.0 * uBusy;
    float wave = sin(dot(position, vec3(9.0, 6.0, 4.0)) - uTime * speed + aRegion * 1.9);
    float pulse = smoothstep(0.6, 1.0, wave) * (0.2 + 0.8 * uBusy) + uLevel * 0.3;

    vec3 col = uColor * (0.22 + 1.1 * lambert * (0.3 + 0.7 * fold) + 0.5 * rim);
    col += uColor * pulse * 1.3 * fold + vec3(0.5) * pulse * pulse * fold;
    // Burst flares in the colour of whoever finished.
    col = mix(col, uBurstColor * (1.3 + rnd.z), k);

    // Hide the far side at rest; every particle shows once it is loose.
    float front = smoothstep(-0.3, 0.25, facing);
    vColor = col;
    vAlpha = mix(front * (0.3 + 0.7 * fold), 1.0, k) * (1.0 - 0.8 * k * smoothstep(0.4, ${HOLD_END.toFixed(2)}, t));
  }
`

const fragment = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    // A soft round glow rather than a square pixel.
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.0, d);
    a *= a;
    gl_FragColor = vec4(vColor * a * vAlpha, 1.0);
  }
`

function useParticles() {
  const [data, setData] = useState<Float32Array | null>(null)
  useEffect(() => {
    let live = true
    fetch(URL)
      .then((r) => r.arrayBuffer())
      .then((buf) => live && setData(new Float32Array(buf)))
      .catch((err) => console.warn('[jarvis] brain particles did not load:', err))
    return () => {
      live = false
    }
  }, [])
  return data
}

export function Brain() {
  const data = useParticles()
  const group = useRef<THREE.Group>(null)
  const size = useThree((s) => s.size)
  const color = useMemo(() => new THREE.Color('#00e5ff'), [])
  const target = useMemo(() => new THREE.Color(), [])

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: vertex,
        fragmentShader: fragment,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        toneMapped: false,
        uniforms: {
          uTime: { value: 0 },
          uBurst: { value: 99 },
          uLevel: { value: 0 },
          uBusy: { value: 0 },
          uViewH: { value: 800 },
          uColor: { value: new THREE.Color('#00e5ff') },
          uBurstColor: { value: new THREE.Color('#00e5ff') },
        },
      }),
    [],
  )

  const geometry = useMemo(() => {
    if (!data) return null
    const count = Math.floor(data.length / STRIDE)
    const pos = new Float32Array(count * 3)
    const nrm = new Float32Array(count * 3)
    const fold = new Float32Array(count)
    const region = new Float32Array(count)
    const seed = new Float32Array(count)
    let minY = Infinity
    let maxY = -Infinity
    for (let i = 0; i < count; i++) {
      const o = i * STRIDE
      pos.set([data[o], data[o + 1], data[o + 2]], i * 3)
      nrm.set([data[o + 3], data[o + 4], data[o + 5]], i * 3)
      fold[i] = data[o + 6]
      region[i] = data[o + 7]
      seed[i] = i * 0.618 + 0.13
      minY = Math.min(minY, data[o + 1])
      maxY = Math.max(maxY, data[o + 1])
    }
    // Centre the brain on the hub.
    const mid = (minY + maxY) / 2
    for (let i = 1; i < pos.length; i += 3) pos[i] -= mid
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('aNormal', new THREE.BufferAttribute(nrm, 3))
    g.setAttribute('aFold', new THREE.BufferAttribute(fold, 1))
    g.setAttribute('aRegion', new THREE.BufferAttribute(region, 1))
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1))
    return g
  }, [data])

  useEffect(() => () => geometry?.dispose(), [geometry])
  useEffect(() => () => material.dispose(), [material])

  useFrame((state, dt) => {
    const { phase, level, ui } = useStore.getState()
    const busy = phase === 'thinking' || phase === 'tooling' || phase === 'speaking'
    target.set(accentFor(phase, ui))
    color.lerp(target, Math.min(1, dt * 3))

    fx.t += dt
    if (fx.t > BACK_END && fx.queue.length) {
      fx.t = 0
      fx.color.set(fx.queue.shift()!)
    }

    const u = material.uniforms
    u.uTime.value = state.clock.elapsedTime
    u.uBurst.value = fx.t
    u.uViewH.value = size.height * state.gl.getPixelRatio()
    u.uLevel.value += (level - u.uLevel.value) * Math.min(1, dt * 8)
    u.uBusy.value += ((busy ? 1 : 0) - u.uBusy.value) * Math.min(1, dt * 2)
    ;(u.uColor.value as THREE.Color).copy(color)
    ;(u.uBurstColor.value as THREE.Color).copy(fx.color)

    // A slow turntable, quicker while thinking, so every side gets seen.
    if (group.current) group.current.rotation.y += dt * (0.18 + 0.45 * u.uBusy.value)
  })

  if (!geometry) return null
  return (
    <group ref={group} scale={1.6}>
      <points geometry={geometry} material={material} frustumCulled={false} />
    </group>
  )
}
