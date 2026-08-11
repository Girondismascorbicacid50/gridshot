import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EdgeSplitModifier } from "three/examples/jsm/modifiers/EdgeSplitModifier.js";

/** three.js preview of the generated bin GLB. Gridfinity is z-up, so the
 *  camera's up vector is set to z and it looks down from a front-iso angle;
 *  drag to orbit, scroll to zoom. Bin rendered in Instrument Teal on Deep Field. */
export function BinViewer({ url }: { url: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const w = el.clientWidth || 600;
    const h = el.clientHeight || 340;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x17191c); // --field
    const camera = new THREE.PerspectiveCamera(35, w / h, 1, 5000);
    camera.up.set(0, 0, 1); // gridfinity z-up
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xf3efe4, 0x243049, 2.0));
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(0.5, -0.8, 1.6);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xf3efe4, 0.8);
    fill.position.set(-1, 0.5, 0.4);
    scene.add(fill);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;

    let raf = 0;
    let disposed = false;
    new GLTFLoader().load(url, (gltf) => {
      if (disposed) return;
      const obj = gltf.scene;
      obj.traverse((c) => {
        const m = c as THREE.Mesh;
        if (m.isMesh) {
          // trimesh GLB ships without normals. A plain computeVertexNormals()
          // averages across every shared vertex, so hard edges (the flat top
          // meeting the vertical pocket wall) shade as a gradient — the "fade"
          // and "canyon" look. Split vertices at a crease angle first: hard
          // edges stay crisp, rounded corners/chamfers stay smooth.
          try {
            const split = new EdgeSplitModifier();
            m.geometry = split.modify(m.geometry, (30 * Math.PI) / 180, false);
          } catch {
            // fall back to smooth normals rather than a blank viewer
          }
          m.geometry.computeVertexNormals();
          m.material = new THREE.MeshStandardMaterial({
            color: 0x2f8f95,
            roughness: 0.6,
            metalness: 0.0,
          });
        }
      });
      const box = new THREE.Box3().setFromObject(obj);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      obj.position.sub(center);
      scene.add(obj);

      const d = Math.max(size.x, size.y, size.z);
      // look down into the pocket (opening is at +z) from a front-iso angle
      camera.position.set(d * 0.55, -d * 0.85, d * 1.15);
      camera.lookAt(0, 0, 0);
      controls.target.set(0, 0, 0);
      controls.update();
    });

    const onResize = () => {
      const W = el.clientWidth || w;
      const H = el.clientHeight || h;
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      renderer.setSize(W, H);
    };
    window.addEventListener("resize", onResize);

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
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === el) el.removeChild(renderer.domElement);
    };
  }, [url]);

  return <div ref={ref} className="w-full h-full" />;
}
