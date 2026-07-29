import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Three.js scene wrapper: a single indexed triangle mesh with static
 * per-vertex positions and dynamic per-vertex colors, orbit controls, and
 * optional camera synchronization with sibling scenes.
 *
 * Adapted from figpack's SphereEmbedding view (figpack_experimental).
 */
export class SphereScene {
  #scene: THREE.Scene;
  #camera: THREE.PerspectiveCamera;
  #renderer: THREE.WebGLRenderer;
  #controls: OrbitControls;
  #geometry: THREE.BufferGeometry;
  #mesh: THREE.Mesh;
  #animationId: number | null = null;
  #defaultCameraState: {
    position: THREE.Vector3;
    target: THREE.Vector3;
  } | null = null;
  #syncing = false;
  #lastW = -1;
  #lastH = -1;

  constructor(
    container: HTMLElement,
    numVertices: number,
    indices: Uint32Array,
    positions: Float32Array,
    background = '#14161c',
  ) {
    this.#scene = new THREE.Scene();
    this.#scene.background = new THREE.Color(background);

    this.#camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);

    this.#renderer = new THREE.WebGLRenderer({ antialias: true });
    this.#renderer.setPixelRatio(window.devicePixelRatio || 1);
    // The canvas always fills its container via CSS; resize() then only
    // updates the drawing buffer
    this.#renderer.domElement.style.width = '100%';
    this.#renderer.domElement.style.height = '100%';
    this.#renderer.domElement.style.display = 'block';
    container.appendChild(this.#renderer.domElement);

    // Lighting: ambient plus a headlight attached to the camera so the
    // surface stays lit from the viewing direction as it is rotated
    this.#scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const headlight = new THREE.DirectionalLight(0xffffff, 1.6);
    headlight.position.set(0.5, 0.8, 1);
    this.#camera.add(headlight);
    this.#scene.add(this.#camera);

    this.#geometry = new THREE.BufferGeometry();
    const positionAttr = new THREE.BufferAttribute(positions, 3);
    const colorAttr = new THREE.BufferAttribute(
      new Float32Array(numVertices * 3),
      3,
    );
    colorAttr.setUsage(THREE.DynamicDrawUsage);
    this.#geometry.setAttribute('position', positionAttr);
    this.#geometry.setAttribute('color', colorAttr);
    this.#geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    this.#geometry.computeVertexNormals();
    this.#geometry.computeBoundingSphere();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      shininess: 25,
      specular: new THREE.Color(0x222222),
    });
    this.#mesh = new THREE.Mesh(this.#geometry, material);
    this.#scene.add(this.#mesh);

    this.#controls = new OrbitControls(this.#camera, this.#renderer.domElement);
    this.#controls.enableDamping = true;
    this.#controls.dampingFactor = 0.1;

    this.#animate();
  }

  #animate = () => {
    this.#animationId = requestAnimationFrame(this.#animate);
    this.#controls.update();
    this.#renderer.render(this.#scene, this.#camera);
  };

  updateColors(colors: Float32Array): void {
    const attr = this.#geometry.getAttribute('color') as THREE.BufferAttribute;
    (attr.array as Float32Array).set(colors);
    attr.needsUpdate = true;
  }

  /** Mirror this scene's camera whenever the other scene's controls move. */
  syncCamerasWith(other: SphereScene): void {
    const follow = (src: SphereScene, dst: SphereScene) => {
      src.#controls.addEventListener('change', () => {
        if (dst.#syncing) return;
        src.#syncing = true;
        dst.#camera.position.copy(src.#camera.position);
        dst.#camera.zoom = src.#camera.zoom;
        dst.#camera.updateProjectionMatrix();
        dst.#controls.target.copy(src.#controls.target);
        dst.#controls.update();
        src.#syncing = false;
      });
    };
    follow(this, other);
    follow(other, this);
  }

  /** Position the camera to comfortably frame the geometry. */
  fitCamera(): void {
    this.#geometry.computeBoundingSphere();
    const bs = this.#geometry.boundingSphere;
    if (!bs) return;
    const radius = Math.max(bs.radius, 1e-6);
    const distance = radius * 2.6;
    this.#controls.target.copy(bs.center);
    this.#camera.position.set(
      bs.center.x + distance * 0.55,
      bs.center.y + distance * 0.35,
      bs.center.z + distance * 0.75,
    );
    this.#camera.near = radius * 0.01;
    this.#camera.far = radius * 100;
    this.#camera.updateProjectionMatrix();
    this.#controls.update();
    this.#defaultCameraState = {
      position: this.#camera.position.clone(),
      target: this.#controls.target.clone(),
    };
  }

  resetCamera(): void {
    if (this.#defaultCameraState) {
      this.#camera.position.copy(this.#defaultCameraState.position);
      this.#controls.target.copy(this.#defaultCameraState.target);
      this.#controls.update();
    } else {
      this.fitCamera();
    }
  }

  resize(width: number, height: number): void {
    // Setting canvas.width clears the canvas even at the same value, which
    // shows as a blank flash until the next render — skip no-op resizes.
    if (width === this.#lastW && height === this.#lastH) return;
    this.#lastW = width;
    this.#lastH = height;
    this.#camera.aspect = width / Math.max(1, height);
    this.#camera.updateProjectionMatrix();
    // updateStyle=false: the canvas keeps its 100%/100% CSS sizing
    this.#renderer.setSize(width, height, false);
  }

  dispose(): void {
    if (this.#animationId !== null) {
      cancelAnimationFrame(this.#animationId);
      this.#animationId = null;
    }
    this.#controls.dispose();
    this.#geometry.dispose();
    (this.#mesh.material as THREE.Material).dispose();
    if (this.#renderer.domElement.parentNode) {
      this.#renderer.domElement.parentNode.removeChild(
        this.#renderer.domElement,
      );
    }
    this.#renderer.dispose();
  }
}
