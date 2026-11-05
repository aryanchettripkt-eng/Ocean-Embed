import {
  WebGLRenderer,
  Scene,
  OrthographicCamera,
  PlaneGeometry,
  BufferGeometry,
  Float32BufferAttribute,
  ShaderMaterial,
  Mesh,
  Points,
  TextureLoader,
  SRGBColorSpace,
  DoubleSide,
  AdditiveBlending,
  type Texture,
} from "three";
import { gsap } from "gsap";
import { fishLayout } from "./fish-layout";
import { sampleSwimRoute, type SwimRoute } from "./ocean-swimming";
import {
  plateVertex,
  waterFragment,
  surfaceVertex,
  surfaceFragment,
  fishVertex,
  fishFragment,
  particleVertex,
  particleFragment,
} from "./ocean-shaders";

export interface OceanScene {
  setPaused(value: boolean): void;
  setTheme(theme: "light" | "dark"): void;
  dispose(): void;
}

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
  const geometries: BufferGeometry[] = [];
  const swimmers: {
    mesh: Mesh;
    material: ShaderMaterial;
    route: SwimRoute;
    progress: { value: number };
    path: gsap.core.Tween;
    offset: number;
    opacity: number;
  }[] = [];
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
  // Fixed viewpoint: neither the camera nor the scene responds to pointer movement.
  const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 3;
  const uniforms = {
    uLight: { value: null as Texture | null },
    uDark: { value: null as Texture | null },
    uTime: { value: 0 },
    uTheme: { value: theme === "dark" ? 1 : 0 },
    uPixelRatio: { value: 1 },
  };
  const draw = () => {
    if (!disposed) renderer.render(scene, camera);
  };
  const updateSwimmers = () => {
    for (const swimmer of swimmers) {
      swimmer.path.totalTime(time + swimmer.offset);
      const pose = sampleSwimRoute(swimmer.progress.value, time, swimmer.route);
      swimmer.mesh.position.x = pose.x;
      swimmer.mesh.position.y = pose.y;
      swimmer.mesh.rotation.z = pose.rotation;
      swimmer.material.uniforms.uOpacity.value = pose.opacity * swimmer.opacity;
    }
  };
  const update = (_elapsed: number, delta: number) => {
    accumulatedDelta += delta;
    if (mobile && accumulatedDelta < 1000 / 30) return;
    time += Math.min(accumulatedDelta, 100) / 1000;
    accumulatedDelta = 0;
    uniforms.uTime.value = time;
    updateSwimmers();
    draw();
  };
  const syncRunning = () => {
    const next = !paused && visible && !document.hidden && !disposed;
    if (next === running) return;
    running = next;
    accumulatedDelta = 0;
    if (running) gsap.ticker.add(update);
    else gsap.ticker.remove(update);
    canvas.dataset.running = String(running);
  };
  const resize = () => {
    const ratio = Math.min(window.devicePixelRatio || 1, mobile ? 1 : 1.5);
    uniforms.uPixelRatio.value = ratio;
    renderer.setPixelRatio(ratio);
    renderer.setSize(host.clientWidth, host.clientHeight, false);
    draw();
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
    swimmers.forEach((swimmer) => swimmer.path.kill());
    observer.disconnect();
    intersection.disconnect();
    document.removeEventListener("visibilitychange", syncRunning);
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
      vertexShader: plateVertex,
      fragmentShader: waterFragment,
      uniforms,
      depthWrite: false,
    });
    materials.push(water);
    const plate = new PlaneGeometry(2, 2);
    geometries.push(plate);
    scene.add(new Mesh(plate, water));

    // A real subdivided ceiling recedes overhead. Vertex waves change its shape and normals.
    const ceilingGeometry = new PlaneGeometry(
      2.12,
      3,
      mobile ? 48 : 96,
      mobile ? 24 : 48,
    );
    geometries.push(ceilingGeometry);
    const ceilingMaterial = new ShaderMaterial({
      vertexShader: surfaceVertex,
      fragmentShader: surfaceFragment,
      uniforms,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      side: DoubleSide,
    });
    materials.push(ceilingMaterial);
    const ceiling = new Mesh(ceilingGeometry, ceilingMaterial);
    ceiling.rotation.x = 1.15;
    ceiling.position.set(0, 0.96, 0.2);
    ceiling.renderOrder = 1;
    ceiling.frustumCulled = false;
    scene.add(ceiling);

    const fishGeometry = new PlaneGeometry(1, 1, 16, 3);
    geometries.push(fishGeometry);
    const core = fishLayout.slice(0, mobile ? 10 : 18).map((fish, i) => ({
      ...fish,
      depth: 0.5 + (i % 4) * 0.09,
      duration: 40 + i * 1.9,
      delay: 0,
    }));
    const distant = Array.from({ length: mobile ? 4 : 8 }, (_, i) => ({
      x: 0.21 + ((i * 0.618) % 1) * 0.65,
      y: 0.64 + Math.sin(i * 2.2) * 0.08,
      width: 0.018 + (i % 3) * 0.004,
      opacity: 0.22 + (i % 3) * 0.05,
      depth: 0.2,
      duration: 75 + i * 4,
      delay: 0,
    }));
    const near = Array.from({ length: mobile ? 1 : 2 }, (_, i) => ({
      x: 0.86 - i * 0.9,
      y: 0.72 + i * 0.1,
      width: 0.105 + i * 0.02,
      opacity: 0.76,
      depth: 1,
      duration: 25 + i * 6,
      delay: 22 + i * 10,
    }));
    [...distant, ...core, ...near].forEach((fish, i) => {
      const route: SwimRoute = {
        direction: fish.depth < 0.3 && i % 3 === 0 ? 1 : -1,
        y: 1 - fish.y * 2,
        bend: Math.sin(i * 2.3) * 0.11,
        phase: i * 2.1,
      };
      const material = new ShaderMaterial({
        vertexShader: fishVertex,
        fragmentShader: fishFragment,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        uniforms: {
          uTime: uniforms.uTime,
          uTheme: uniforms.uTheme,
          uFish: { value: fishTexture },
          uPhase: { value: route.phase },
          uSpeed: { value: 3.6 + (i % 5) * 0.46 },
          uOpacity: { value: 0 },
          uDepth: { value: fish.depth },
        },
      });
      materials.push(material);
      const mesh = new Mesh(fishGeometry, material);
      mesh.scale.set(
        -route.direction * fish.width * 2,
        (fish.width * 2 * (1672 / 941)) / 3,
        1,
      );
      mesh.position.z = 0.1 + fish.depth * 0.1;
      mesh.renderOrder = 2 + Math.round(fish.depth * 2);
      scene.add(mesh);
      const progress = { value: 0 };
      const path = gsap.to(progress, {
        value: 1,
        duration: fish.duration,
        repeat: -1,
        repeatDelay: fish.delay,
        ease: "none",
        paused: true,
      });
      const offset =
        ((1.325 + route.direction * (fish.x * 2 - 1)) / 2.65) * fish.duration;
      swimmers.push({
        mesh,
        material,
        route,
        progress,
        path,
        offset,
        opacity: fish.opacity,
      });
    });
    updateSwimmers();

    const count = mobile ? 28 : 56;
    const positions: number[] = [],
      seeds: number[] = [];
    for (let i = 0; i < count; i++) {
      positions.push(
        ((i * 0.6180339) % 1) * 2 - 1,
        ((i * 0.4142135) % 1) * 2 - 1,
        0.3,
      );
      seeds.push((i * 0.73205) % 1);
    }
    const particlesGeometry = new BufferGeometry();
    geometries.push(particlesGeometry);
    particlesGeometry.setAttribute(
      "position",
      new Float32BufferAttribute(positions, 3),
    );
    particlesGeometry.setAttribute(
      "aSeed",
      new Float32BufferAttribute(seeds, 1),
    );
    const particleMaterial = new ShaderMaterial({
      vertexShader: particleVertex,
      fragmentShader: particleFragment,
      uniforms,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: AdditiveBlending,
    });
    materials.push(particleMaterial);
    const particles = new Points(particlesGeometry, particleMaterial);
    particles.renderOrder = 5;
    particles.frustumCulled = false;
    scene.add(particles);

    await renderer.compileAsync(scene, camera);
    if (disposed) return null;
    host.appendChild(canvas);
    resize();
    observer.observe(host);
    intersection.observe(host);
    document.addEventListener("visibilitychange", syncRunning);
    syncRunning();
    return {
      setPaused(value) {
        paused = value;
        syncRunning();
      },
      setTheme(value) {
        themeTween?.kill();
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
