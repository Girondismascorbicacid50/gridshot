import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EdgeSplitModifier } from "three/examples/jsm/modifiers/EdgeSplitModifier.js";

function objectPath(object: THREE.Object3D): string {
  const names: string[] = [];
  let current: THREE.Object3D | null = object;
  while (current) {
    if (current.name) names.push(current.name);
    current = current.parent;
  }
  return names.join("/");
}

/** Orbitable GLB scene for a complete drawer composition. The backend scene
 * contains the real Gridfinity socket grid and each exact regenerated bin. */
export function DrawerViewer({
  url,
  binColors,
}: {
  url: string;
  binColors: Record<string, string>;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const width = el.clientWidth || 600;
    const height = el.clientHeight || 380;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x17191c);
    const camera = new THREE.PerspectiveCamera(35, width / height, 1, 6000);
    camera.up.set(0, 0, 1);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xf3efe4, 0x172033, 2.2));
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const key = new THREE.DirectionalLight(0xffffff, 2.5);
    key.position.set(0.6, -0.9, 1.8);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x8ed8db, 0.9);
    fill.position.set(-1, 0.5, 0.7);
    scene.add(fill);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;

    let raf = 0;
    let disposed = false;
    let loadedObject: THREE.Object3D | null = null;
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (disposed) return;
        const object = gltf.scene;
        object.traverse((child) => {
          const mesh = child as THREE.Mesh;
          if (!mesh.isMesh) return;
          try {
            mesh.geometry = new EdgeSplitModifier().modify(
              mesh.geometry,
              (30 * Math.PI) / 180,
              false,
            );
          } catch {
            // Smooth normals are still preferable to dropping the scene.
          }
          mesh.geometry.computeVertexNormals();
          const path = objectPath(mesh);
          const toolId = Object.keys(binColors).find((id) =>
            path.includes(`drawer-bin-${id}`),
          );
          const isGrid = path.includes("drawer-grid");
          const originalMaterials = Array.isArray(mesh.material)
            ? mesh.material
            : [mesh.material];
          originalMaterials.forEach((material) => material.dispose());
          mesh.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(
              toolId ? binColors[toolId] : isGrid ? "#39475d" : "#2f8f95",
            ),
            roughness: isGrid ? 0.82 : 0.58,
            metalness: 0,
          });
        });

        const bounds = new THREE.Box3().setFromObject(object);
        const size = bounds.getSize(new THREE.Vector3());
        const center = bounds.getCenter(new THREE.Vector3());
        object.position.sub(center);
        scene.add(object);
        loadedObject = object;

        const span = Math.max(size.x, size.y, size.z * 3, 1);
        camera.position.set(span * 0.72, -span * 0.9, span * 0.95);
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();
      },
      undefined,
      () => {
        // The caller exposes request failures; a malformed GLB simply leaves
        // the preview background visible instead of crashing the page.
      },
    );

    const resize = () => {
      const nextWidth = el.clientWidth || width;
      const nextHeight = el.clientHeight || height;
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    };
    window.addEventListener("resize", resize);

    const loop = () => {
      if (disposed) return;
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      controls.dispose();
      loadedObject?.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh) return;
        mesh.geometry.dispose();
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        materials.forEach((material) => material.dispose());
      });
      renderer.dispose();
      if (renderer.domElement.parentNode === el) el.removeChild(renderer.domElement);
    };
  }, [url, binColors]);

  return <div ref={ref} className="h-full w-full" />;
}
