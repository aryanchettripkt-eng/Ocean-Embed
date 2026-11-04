import {
  WebGLRenderer,
  Scene,
  OrthographicCamera,
  PlaneGeometry,
  ShaderMaterial,
  Mesh,
  TextureLoader,
  SRGBColorSpace,
  Vector2,
  type Texture,
} from "three";
import { gsap } from "gsap";
import { fishLayout } from "./fish-layout";

export interface OceanScene {
  setPaused(value: boolean): void;
  setTheme(theme: "light" | "dark"): void;
  dispose(): void;
}
const vertex = `varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }`;
const waterFragment = `
uniform sampler2D uLight; uniform sampler2D uDark;
uniform float uTime; uniform float uTheme; uniform vec2 uPointer;
varying vec2 vUv;
void main() {
  vec2 uv = vUv;
  float surface = smoothstep(.69, 1., uv.y);
  // Refraction is confined to the surface: the reef and tether stay anchored.
  uv.x += surface * (sin(uv.y*69. + uTime*.71) + sin(uv.x*51.-uTime*.48))*.0028;
  uv.y += surface * sin(uv.x*78. + uTime*.83)*.0022;
  uv += uPointer * surface * .001;
  vec3 base = mix(texture2D(uLight, uv).rgb, texture2D(uDark, uv).rgb, uTheme);
  float ripples = sin(uv.x*128. + sin(uv.y*71.-uTime*.45)*2. + uTime*.6);
  float highlight = smoothstep(.16,.6,max(base.r,max(base.g,base.b)));
  float shimmer = pow(max(0., ripples), 12.) * surface * highlight;
  float shaft = 0.;
  for (int i=0; i<5; i++) {
    float n = float(i);
    float origin = .49 + n*.073 + sin(uTime*.13+n*2.)*.012;
    float ray = uv.x - origin + (1.-uv.y)*(.22+n*.025);
    shaft += exp(-ray*ray/(.00025+(1.-uv.y)*.0013)) * (.55+.45*sin(uTime*.21+n));
  }
  shaft *= smoothstep(.08, .85, uv.y)*(1.-smoothstep(.87,1.,uv.y));
  float floorMask = (1.-smoothstep(.08,.4,uv.y))*smoothstep(.08,.38,uv.x);
  float caustic = pow(max(0.,sin(uv.x*85.+sin(uv.y*43.+uTime*.3)*2.-uTime*.37)
    *sin(uv.y*71.-sin(uv.x*48.-uTime*.26)*2.+uTime*.41)), 5.);
  vec3 light = mix(vec3(1.,.79,.43), vec3(.23,.83,.88), uTheme);
  base += light * (shimmer*.018 + shaft*.006 + caustic*floorMask*.003);
  gl_FragColor = vec4(base, 1.);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;
const fishVertex = `uniform float uTime; uniform float uPhase; varying vec2 vUv;
void main() {
 vUv=uv; vec3 p=position;
 // Head is left; flex progressively toward the forked tail on the right.
 p.y += sin(uTime*3.2+uPhase+uv.x*5.)*pow(uv.x,3.)*.075;
 p.x *= 1.-pow(uv.x,2.)*(.025+.025*sin(uTime*3.2+uPhase));
 gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);
}`;
const fishFragment = `uniform sampler2D uFish; uniform float uOpacity; uniform float uTheme; varying vec2 vUv;
void main() { vec4 c=texture2D(uFish,vUv); if(c.a<.02) discard;
 c.rgb *= mix(.68,.38,uTheme); gl_FragColor=vec4(c.rgb,c.a*uOpacity);
 #include <tonemapping_fragment>
 #include <colorspace_fragment>
}`;

export async function createOceanScene(
  host: HTMLElement,
  theme: "light" | "dark",
  onFailure: () => void,
  signal?: AbortSignal,
): Promise<OceanScene | null> {
  if (signal?.aborted) return null;
  let renderer: WebGLRenderer;
  try {
    renderer = new WebGLRenderer({
      alpha: false,
      antialias: false,
      powerPreference: "low-power",
    });
  } catch {
    return null;
  }
  const mobile = window.matchMedia("(max-width: 767px)").matches;
  const loader = new TextureLoader();
  const textures: Texture[] = [];
  const materials: ShaderMaterial[] = [];
  const geometries: PlaneGeometry[] = [];
  const paths: gsap.core.Tween[] = [];
  let disposed = false;
  let paused = false;
  let visible = true;
  let running = false;
  let time = 0;
  let accumulatedDelta = 0;
  let themeTween: gsap.core.Tween | undefined;
  const canvas = renderer.domElement;
  canvas.dataset.oceanCanvas = "true";
  canvas.setAttribute("aria-hidden", "true");
  const scene = new Scene();
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 2;
  const uniforms = {
    uLight: { value: null as Texture | null },
    uDark: { value: null as Texture | null },
    uTime: { value: 0 },
    uTheme: { value: theme === "dark" ? 1 : 0 },
    uPointer: { value: new Vector2() },
  };
  const draw = () => {
    if (!disposed) renderer.render(scene, camera);
  };
  const update = (_elapsed: number, delta: number) => {
    accumulatedDelta += delta;
    if (mobile && accumulatedDelta < 1000 / 30) return;
    time += Math.min(accumulatedDelta, 100) / 1000;
    accumulatedDelta = 0;
    uniforms.uTime.value = time;
    for (const material of materials)
      if (material.uniforms.uTime) material.uniforms.uTime.value = time;
    for (const path of paths) path.totalTime(time);
    draw();
  };
  const syncRunning = () => {
    const next = !paused && visible && !document.hidden && !disposed;
    if (next === running) return;
    running = next;
    if (running) gsap.ticker.add(update);
    else gsap.ticker.remove(update);
    canvas.dataset.running = String(running);
  };
  const resize = () => {
    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio || 1, mobile ? 1 : 1.5),
    );
    renderer.setSize(host.clientWidth, host.clientHeight, false);
    draw();
  };
  const pointer = { x: 0, y: 0 };
  const move = (event: PointerEvent) => {
    if (paused || event.pointerType !== "mouse") return;
    const rect = host.getBoundingClientRect();
    gsap.to(pointer, {
      x: (event.clientX - rect.left) / rect.width - 0.5,
      y: (event.clientY - rect.top) / rect.height - 0.5,
      duration: 1.5,
      overwrite: true,
      onUpdate: () => uniforms.uPointer.value.set(pointer.x, pointer.y),
    });
  };
  const lost = (event: Event) => {
    event.preventDefault();
    dispose();
    onFailure();
  };
  const observer = new ResizeObserver(resize);
  const intersection = new IntersectionObserver((entries) => {
    visible = entries[0]?.isIntersecting ?? true;
    syncRunning();
  });
  function dispose() {
    if (disposed) return;
    disposed = true;
    gsap.ticker.remove(update);
    themeTween?.kill();
    gsap.killTweensOf(pointer);
    paths.forEach((path) => path.kill());
    observer.disconnect();
    intersection.disconnect();
    document.removeEventListener("visibilitychange", syncRunning);
    window.removeEventListener("pointermove", move);
    canvas.removeEventListener("webglcontextlost", lost);
    signal?.removeEventListener("abort", dispose);
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    textures.forEach((texture) => texture.dispose());
    renderer.dispose();
    canvas.remove();
  }
  canvas.addEventListener("webglcontextlost", lost);
  signal?.addEventListener("abort", dispose, { once: true });
  try {
    // All settled: even a successful sibling load is disposed if another texture fails.
    const suffix = mobile ? 1024 : 1600;
    const loaded = await Promise.allSettled(
      [
        `/landing/ocean/light-${suffix}.webp`,
        `/landing/ocean/dark-${suffix}.webp`,
        "/landing/ocean/fish.webp",
      ].map(async (url) => {
        const texture = await loader.loadAsync(url);
        if (disposed) {
          texture.dispose();
          return texture;
        }
        texture.colorSpace = SRGBColorSpace;
        textures.push(texture);
        return texture;
      }),
    );
    if (disposed) return null;
    if (loaded.some((result) => result.status === "rejected")) {
      dispose();
      return null;
    }
    const [lightMap, darkMap, fishTexture] = loaded.flatMap((result) =>
      result.status === "fulfilled" ? [result.value] : [],
    );
    uniforms.uLight.value = lightMap;
    uniforms.uDark.value = darkMap;
    const water = new ShaderMaterial({
      vertexShader: vertex,
      fragmentShader: waterFragment,
      uniforms,
      depthWrite: false,
    });
    materials.push(water);
    const plane = new PlaneGeometry(2, 2);
    geometries.push(plane);
    scene.add(new Mesh(plane, water));
    const fishGeometry = new PlaneGeometry(1, 1, 12, 2);
    geometries.push(fishGeometry);
    fishLayout.slice(0, mobile ? 10 : 18).forEach((fish, i) => {
      const material = new ShaderMaterial({
        vertexShader: fishVertex,
        fragmentShader: fishFragment,
        transparent: true,
        depthWrite: false,
        uniforms: {
          uTime: { value: 0 },
          uPhase: { value: i * 2.1 },
          uFish: { value: fishTexture },
          uOpacity: { value: fish.opacity },
          uTheme: uniforms.uTheme,
        },
      });
      materials.push(material);
      const mesh = new Mesh(fishGeometry, material);
      mesh.scale.set(fish.width * 2, (fish.width * 2 * (1672 / 941)) / 3, 1);
      mesh.position.set(fish.x * 2 - 1, 1 - fish.y * 2, 0.1 + i * 0.001);
      mesh.rotation.z = -0.08;
      scene.add(mesh);
      paths.push(
        gsap.to(mesh.position, {
          x: mesh.position.x - 0.48,
          y: mesh.position.y + 0.06,
          duration: 32 + i * 2.3,
          repeat: -1,
          yoyo: true,
          ease: "sine.inOut",
          paused: true,
        }),
      );
    });
    // Parallel GPU compilation avoids a synchronous shader stall on the first frame.
    await renderer.compileAsync(scene, camera);
    if (disposed) return null;
    host.appendChild(canvas);
    resize();
    observer.observe(host);
    intersection.observe(host);
    document.addEventListener("visibilitychange", syncRunning);
    window.addEventListener("pointermove", move, { passive: true });
    syncRunning();
    return {
      setPaused(value) {
        paused = value;
        if (value) gsap.killTweensOf(pointer);
        syncRunning();
      },
      setTheme(value) {
        themeTween?.kill();
        // User pause freezes the scene, but an intentional theme change still renders.
        themeTween = gsap.to(uniforms.uTheme, {
          value: value === "dark" ? 1 : 0,
          duration: paused ? 0 : 0.65,
          onUpdate: draw,
        });
      },
      dispose,
    };
  } catch {
    dispose();
    return null;
  }
}
