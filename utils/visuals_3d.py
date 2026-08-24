"""3D Visualizations Engine for AI Resume Analyzer.

Provides interactive WebGL/Three.js 3D components and Plotly 3D charts:
1. Three.js 3D Interactive Cyber Wave & Particle Hero Canvas
2. Three.js 3D Holographic ATS Compatibility Score Orb & Energy Rings
3. Three.js 3D Interactive Skill Constellation & Orbital Node Graph
4. Three.js 3D Quantum Login Portal & Particle Vortex
5. Plotly 3D Semantic Vector Space (Resume vs JD vs Benchmark Roles)
6. Plotly 3D Capability Surface / Radar Mesh
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from sklearn.decomposition import TruncatedSVD
from utils.text_processing import clean_text


def get_3d_login_portal_html() -> str:
    """Generates an embedded Three.js WebGL canvas displaying a 3D Quantum Portal and particle vortex for the login page."""
    return """
    <div style="position: relative; width: 100%; height: 380px; border-radius: 20px; overflow: hidden; background: radial-gradient(circle at 50% 50%, #0d1527 0%, #03050a 100%); box-shadow: 0 20px 60px rgba(0, 245, 212, 0.15), inset 0 0 40px rgba(0, 0, 0, 0.9); border: 1px solid rgba(0, 245, 212, 0.3); display: flex; flex-direction: column; align-items: center; justify-content: center;">
        
        <div id="three-login-portal" style="width: 100%; height: 100%; position: absolute; top:0; left:0; z-index: 1;"></div>
        
        <div style="position: relative; z-index: 2; text-align: center; pointer-events: none; padding: 0 20px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(0, 245, 212, 0.15); border: 1px solid rgba(0, 245, 212, 0.5); border-radius: 20px; padding: 4px 14px; margin-bottom: 12px; backdrop-filter: blur(10px);">
                <span style="width: 8px; height: 8px; background: #00F5D4; border-radius: 50%; box-shadow: 0 0 10px #00F5D4;"></span>
                <span style="color: #00F5D4; font-family: 'Segoe UI', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;">Secure Authentication Portal</span>
            </div>
            
            <h1 style="margin: 0; font-family: 'Montserrat', sans-serif; font-size: 38px; font-weight: 900; background: linear-gradient(135deg, #FFFFFF 30%, #00F5D4 70%, #7B2CBF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 1px; text-shadow: 0 0 35px rgba(0, 245, 212, 0.4);">
                AI RESUME ANALYZER 3D
            </h1>
            <p style="margin: 6px 0 0 0; color: #94A3B8; font-family: 'Segoe UI', sans-serif; font-size: 14px; font-weight: 400; max-width: 480px; text-shadow: 0 2px 4px rgba(0,0,0,0.8);">
                Sign in to access your 3D ATS Compatibility Engine, ML Category Classifier, and Skill Constellations.
            </p>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {
        const container = document.getElementById('three-login-portal');
        if (!container) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(0, 0, 28);

        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const portalGroup = new THREE.Group();
        scene.add(portalGroup);

        // Nested Quantum Rings
        const ringMat1 = new THREE.MeshBasicMaterial({ color: 0x00F5D4, wireframe: true, transparent: true, opacity: 0.35 });
        const ringMat2 = new THREE.MeshBasicMaterial({ color: 0x7B2CBF, wireframe: true, transparent: true, opacity: 0.4 });
        const ringMat3 = new THREE.MeshBasicMaterial({ color: 0x00BBF9, wireframe: true, transparent: true, opacity: 0.3 });

        const ring1 = new THREE.Mesh(new THREE.TorusGeometry(10, 0.3, 16, 60), ringMat1);
        const ring2 = new THREE.Mesh(new THREE.TorusGeometry(8.2, 0.25, 16, 50), ringMat2);
        const ring3 = new THREE.Mesh(new THREE.TorusGeometry(6.4, 0.2, 16, 40), ringMat3);
        
        portalGroup.add(ring1);
        portalGroup.add(ring2);
        portalGroup.add(ring3);

        // Core Quantum Crystal
        const coreGeo = new THREE.OctahedronGeometry(3.5, 0);
        const coreMat = new THREE.MeshBasicMaterial({ color: 0x00F5D4, wireframe: true, transparent: true, opacity: 0.5 });
        const coreCrystal = new THREE.Mesh(coreGeo, coreMat);
        portalGroup.add(coreCrystal);

        // Particle Vortex Stream
        const particleCount = 200;
        const pGeo = new THREE.BufferGeometry();
        const pPos = new Float32Array(particleCount * 3);
        const pColors = new Float32Array(particleCount * 3);

        for (let i = 0; i < particleCount * 3; i += 3) {
            const angle = (i / (particleCount * 3)) * Math.PI * 8;
            const radius = 4 + Math.random() * 10;
            pPos[i] = Math.cos(angle) * radius;
            pPos[i + 1] = Math.sin(angle) * radius;
            pPos[i + 2] = (Math.random() - 0.5) * 12;

            pColors[i] = 0.0;
            pColors[i + 1] = 0.95;
            pColors[i + 2] = 0.85;
        }

        pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
        pGeo.setAttribute('color', new THREE.BufferAttribute(pColors, 3));

        const pMat = new THREE.PointsMaterial({ size: 0.5, vertexColors: true, transparent: true, opacity: 0.75 });
        const particles = new THREE.Points(pGeo, pMat);
        portalGroup.add(particles);

        let mouseX = 0, mouseY = 0;
        window.addEventListener('mousemove', function(e) {
            mouseX = (e.clientX - window.innerWidth / 2) * 0.005;
            mouseY = (e.clientY - window.innerHeight / 2) * 0.005;
        });

        function animate() {
            requestAnimationFrame(animate);

            ring1.rotation.z += 0.008;
            ring1.rotation.x += 0.004;

            ring2.rotation.z -= 0.01;
            ring2.rotation.y += 0.006;

            ring3.rotation.z += 0.012;
            ring3.rotation.x -= 0.005;

            coreCrystal.rotation.x += 0.015;
            coreCrystal.rotation.y += 0.02;

            particles.rotation.z += 0.006;

            portalGroup.rotation.y += (mouseX - portalGroup.rotation.y) * 0.05;
            portalGroup.rotation.x += (mouseY - portalGroup.rotation.x) * 0.05;

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', function() {
            if (!container) return;
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    })();
    </script>
    """


def get_3d_hero_header_html() -> str:
    """Generates an embedded Three.js WebGL canvas displaying a glowing 3D particle landscape."""
    return """
    <div style="position: relative; width: 100%; height: 250px; border-radius: 16px; overflow: hidden; background: radial-gradient(circle at 50% 50%, #0d1322 0%, #05070e 100%); box-shadow: 0 10px 40px rgba(0, 245, 212, 0.12), inset 0 0 30px rgba(0, 0, 0, 0.8); border: 1px solid rgba(0, 245, 212, 0.25); margin-bottom: 20px;">
        <div id="three-hero-container" style="width: 100%; height: 100%; position: absolute; top:0; left:0; z-index: 1;"></div>
        
        <div style="position: relative; z-index: 2; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; pointer-events: none; padding: 0 20px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(0, 245, 212, 0.12); border: 1px solid rgba(0, 245, 212, 0.4); border-radius: 20px; padding: 4px 14px; margin-bottom: 8px; backdrop-filter: blur(8px);">
                <span style="width: 8px; height: 8px; background: #00F5D4; border-radius: 50%; box-shadow: 0 0 10px #00F5D4;"></span>
                <span style="color: #00F5D4; font-family: 'Segoe UI', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;">Next-Gen AI & 3D Intelligence</span>
            </div>
            
            <h1 style="margin: 0; font-family: 'Segoe UI', 'Montserrat', sans-serif; font-size: 32px; font-weight: 900; background: linear-gradient(135deg, #FFFFFF 20%, #00F5D4 60%, #7B2CBF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 1.5px; text-shadow: 0 0 30px rgba(0, 245, 212, 0.3);">
                AI RESUME ANALYZER 3D
            </h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-family: 'Segoe UI', sans-serif; font-size: 13px; font-weight: 400; max-width: 600px; text-shadow: 0 2px 4px rgba(0,0,0,0.8);">
                Deep Learning ATS Optimization • 3D Skill Constellations • Multi-factor Compatibility
            </p>
        </div>
    </div>

    <!-- Three.js CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {
        const container = document.getElementById('three-hero-container');
        if (!container) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 1, 1000);
        camera.position.set(0, 25, 45);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        // 3D Particle Grid Wave
        const SEPARATION = 4, AMOUNTX = 40, AMOUNTY = 40;
        const numParticles = AMOUNTX * AMOUNTY;
        const positions = new Float32Array(numParticles * 3);
        const scales = new Float32Array(numParticles);
        const colors = new Float32Array(numParticles * 3);

        let i = 0, j = 0;
        for (let ix = 0; ix < AMOUNTX; ix++) {
            for (let iy = 0; iy < AMOUNTY; iy++) {
                positions[i] = ix * SEPARATION - ((AMOUNTX * SEPARATION) / 2);
                positions[i + 1] = 0;
                positions[i + 2] = iy * SEPARATION - ((AMOUNTY * SEPARATION) / 2);

                const ratio = ix / AMOUNTX;
                colors[i] = 0.0 + 0.5 * ratio;
                colors[i + 1] = 0.95 - 0.7 * ratio;
                colors[i + 2] = 0.85 + 0.15 * ratio;

                scales[j] = 1;
                i += 3;
                j++;
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 2.2,
            vertexColors: true,
            transparent: true,
            opacity: 0.85
        });

        const particles = new THREE.Points(geometry, material);
        scene.add(particles);

        // Floating 3D Wireframes
        const polyGeo = new THREE.IcosahedronGeometry(8, 1);
        const polyMat = new THREE.MeshBasicMaterial({
            color: 0x00F5D4,
            wireframe: true,
            transparent: true,
            opacity: 0.35
        });
        const polyhedron = new THREE.Mesh(polyGeo, polyMat);
        polyhedron.position.set(28, 5, -10);
        scene.add(polyhedron);

        const polyGeo2 = new THREE.TorusGeometry(6, 0.4, 16, 50);
        const polyMat2 = new THREE.MeshBasicMaterial({
            color: 0x7B2CBF,
            wireframe: true,
            transparent: true,
            opacity: 0.4
        });
        const torus = new THREE.Mesh(polyGeo2, polyMat2);
        torus.position.set(-28, 4, -10);
        scene.add(torus);

        let count = 0;
        let mouseX = 0;

        window.addEventListener('mousemove', function(event) {
            mouseX = (event.clientX - window.innerWidth / 2) * 0.02;
        });

        function animate() {
            requestAnimationFrame(animate);
            count += 0.04;

            const positionAttribute = geometry.attributes.position;
            let idx = 0;
            for (let ix = 0; ix < AMOUNTX; ix++) {
                for (let iy = 0; iy < AMOUNTY; iy++) {
                    positionAttribute.array[idx + 1] = (Math.sin((ix + count) * 0.3) * 3.5) + (Math.sin((iy + count) * 0.5) * 3.5);
                    idx += 3;
                }
            }
            positionAttribute.needsUpdate = true;

            polyhedron.rotation.x += 0.008;
            polyhedron.rotation.y += 0.012;
            torus.rotation.x += 0.01;
            torus.rotation.y += 0.007;

            camera.position.x += (mouseX - camera.position.x) * 0.05;
            camera.lookAt(0, 0, 0);

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', function() {
            if (!container) return;
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    })();
    </script>
    """


def get_3d_score_orb_html(ats_score: int, rating_label: str, rating_color: str, rating_grade: str) -> str:
    """Generates an embedded Three.js WebGL 3D Holographic Score Orb with orbiting energy rings."""
    template = """
    <div style="position: relative; width: 100%; height: 320px; border-radius: 16px; background: radial-gradient(circle at 50% 50%, #0e1628 0%, #04060c 100%); border: 1px solid rgba(0, 245, 212, 0.3); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6), inset 0 0 30px rgba(0, 245, 212, 0.08); overflow: hidden; display: flex; align-items: center; justify-content: center;">
        
        <div id="three-score-orb-container" style="width: 100%; height: 100%; position: absolute; top:0; left:0; z-index: 1;"></div>
        
        <div style="position: relative; z-index: 2; text-align: center; pointer-events: none; backdrop-filter: blur(2px);">
            <div style="font-family: 'Segoe UI', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 2px; color: #94A3B8; text-transform: uppercase; margin-bottom: 2px;">
                ATS COMPATIBILITY
            </div>
            
            <div style="font-family: 'Montserrat', 'Segoe UI', sans-serif; font-size: 64px; font-weight: 900; line-height: 1; color: #FFFFFF; text-shadow: 0 0 25px __RATING_COLOR__, 0 0 50px __RATING_COLOR__66;">
                __SCORE__<span style="font-size: 32px; color: __RATING_COLOR__;">%</span>
            </div>
            
            <div style="margin-top: 8px; display: inline-flex; align-items: center; gap: 6px; background: rgba(0, 0, 0, 0.6); border: 1px solid __RATING_COLOR__; border-radius: 20px; padding: 4px 14px; box-shadow: 0 0 15px __RATING_COLOR__44;">
                <span style="font-size: 12px; font-weight: 800; color: __RATING_COLOR__; font-family: 'Segoe UI', sans-serif;">
                    GRADE __GRADE__ • __LABEL__
                </span>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {
        const container = document.getElementById('three-score-orb-container');
        if (!container) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.z = 24;

        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const hexColor = parseInt("__RATING_COLOR__".replace("#", "0x"), 16) || 0x00F5D4;
        
        const sphereGeo = new THREE.SphereGeometry(6.5, 24, 24);
        const sphereMat = new THREE.MeshBasicMaterial({
            color: hexColor,
            wireframe: true,
            transparent: true,
            opacity: 0.25
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        scene.add(sphere);

        const ringGeo1 = new THREE.TorusGeometry(8.8, 0.12, 16, 100);
        const ringMat1 = new THREE.MeshBasicMaterial({
            color: hexColor,
            transparent: true,
            opacity: 0.8
        });
        const ring1 = new THREE.Mesh(ringGeo1, ringMat1);
        ring1.rotation.x = Math.PI / 3;
        scene.add(ring1);

        const ringGeo2 = new THREE.TorusGeometry(9.8, 0.08, 16, 100);
        const ringMat2 = new THREE.MeshBasicMaterial({
            color: 0x7B2CBF,
            transparent: true,
            opacity: 0.6
        });
        const ring2 = new THREE.Mesh(ringGeo2, ringMat2);
        ring2.rotation.y = Math.PI / 4;
        scene.add(ring2);

        const particleCount = 100;
        const pGeo = new THREE.BufferGeometry();
        const pPos = new Float32Array(particleCount * 3);
        for (let i = 0; i < particleCount * 3; i += 3) {
            const r = 7.5 + Math.random() * 4.5;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos((Math.random() * 2) - 1);
            pPos[i] = r * Math.sin(phi) * Math.cos(theta);
            pPos[i + 1] = r * Math.sin(phi) * Math.sin(theta);
            pPos[i + 2] = r * Math.cos(phi);
        }
        pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
        const pMat = new THREE.PointsMaterial({
            color: hexColor,
            size: 0.45,
            transparent: true,
            opacity: 0.8
        });
        const particleSystem = new THREE.Points(pGeo, pMat);
        scene.add(particleSystem);

        let mouseX = 0, mouseY = 0;
        container.addEventListener('mousemove', function(e) {
            const rect = container.getBoundingClientRect();
            mouseX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
            mouseY = ((e.clientY - rect.top) / rect.height - 0.5) * 2;
        });

        function animate() {
            requestAnimationFrame(animate);

            sphere.rotation.y += 0.008;
            sphere.rotation.x += 0.004;

            ring1.rotation.z += 0.012;
            ring1.rotation.x += 0.006;

            ring2.rotation.z -= 0.009;
            ring2.rotation.y += 0.008;

            particleSystem.rotation.y += 0.005;
            particleSystem.rotation.x -= 0.003;

            camera.position.x += (mouseX * 3 - camera.position.x) * 0.05;
            camera.position.y += (-mouseY * 3 - camera.position.y) * 0.05;
            camera.lookAt(0, 0, 0);

            renderer.render(scene, camera);
        }
        animate();
    })();
    </script>
    """
    html = template.replace("__SCORE__", str(ats_score))
    html = html.replace("__RATING_COLOR__", rating_color)
    html = html.replace("__LABEL__", rating_label)
    html = html.replace("__GRADE__", rating_grade)
    return html


def get_3d_skill_constellation_html(
    matched_skills: List[str],
    missing_skills: List[str],
    additional_skills: List[str]
) -> str:
    """Generates an interactive 3D WebGL Orbital Constellation Graph of Matched, Missing, and Bonus skills."""
    nodes_data = []
    
    nodes_data.append({"id": "center_resume", "label": "RESUME CORE", "type": "core_resume", "color": "#00F5D4", "size": 3.0, "x": -6, "y": 0, "z": 0})
    nodes_data.append({"id": "center_jd", "label": "JOB DESC CORE", "type": "core_jd", "color": "#7B2CBF", "size": 3.0, "x": 6, "y": 0, "z": 0})

    for i, skill in enumerate(matched_skills[:12]):
        angle = (i / max(len(matched_skills[:12]), 1)) * 2 * np.pi
        r = 3.5 + (i % 3) * 0.8
        nodes_data.append({
            "id": f"matched_{i}",
            "label": f"✓ {skill}",
            "type": "matched",
            "color": "#00F5D4",
            "size": 1.6,
            "x": round(float(np.sin(angle) * r), 2),
            "y": round(float(np.cos(angle) * (r * 0.8)), 2),
            "z": round(float((i % 5 - 2) * 1.5), 2)
        })

    for i, skill in enumerate(missing_skills[:10]):
        angle = (i / max(len(missing_skills[:10]), 1)) * 2 * np.pi
        r = 5.0 + (i % 3) * 0.9
        nodes_data.append({
            "id": f"missing_{i}",
            "label": f"✗ {skill}",
            "type": "missing",
            "color": "#FF007F",
            "size": 1.4,
            "x": round(float(6 + np.sin(angle) * r), 2),
            "y": round(float(np.cos(angle) * r), 2),
            "z": round(float((i % 4 - 2) * 1.8), 2)
        })

    for i, skill in enumerate(additional_skills[:8]):
        angle = (i / max(len(additional_skills[:8]), 1)) * 2 * np.pi
        r = 5.0 + (i % 3) * 0.9
        nodes_data.append({
            "id": f"additional_{i}",
            "label": f"+ {skill}",
            "type": "additional",
            "color": "#00BBF9",
            "size": 1.2,
            "x": round(float(-6 + np.sin(angle) * r), 2),
            "y": round(float(np.cos(angle) * r), 2),
            "z": round(float((i % 4 - 2) * 1.8), 2)
        })

    nodes_json = json.dumps(nodes_data)

    template = """
    <div style="position: relative; width: 100%; height: 420px; border-radius: 16px; background: radial-gradient(circle at 50% 50%, #0d1424 0%, #03060d 100%); border: 1px solid rgba(0, 245, 212, 0.25); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8); overflow: hidden;">
        
        <div style="position: absolute; top: 12px; left: 16px; z-index: 10; display: flex; flex-wrap: wrap; gap: 10px; font-family: 'Segoe UI', sans-serif; font-size: 11px; pointer-events: none;">
            <div style="display: inline-flex; align-items: center; gap: 5px; background: rgba(0, 0, 0, 0.7); padding: 4px 10px; border-radius: 12px; border: 1px solid #00F5D4;">
                <span style="width: 8px; height: 8px; background: #00F5D4; border-radius: 50%; box-shadow: 0 0 6px #00F5D4;"></span>
                <span style="color: #E2E8F0; font-weight: 600;">Matched (__MATCHED_LEN__)</span>
            </div>
            <div style="display: inline-flex; align-items: center; gap: 5px; background: rgba(0, 0, 0, 0.7); padding: 4px 10px; border-radius: 12px; border: 1px solid #FF007F;">
                <span style="width: 8px; height: 8px; background: #FF007F; border-radius: 50%; box-shadow: 0 0 6px #FF007F;"></span>
                <span style="color: #E2E8F0; font-weight: 600;">Missing (__MISSING_LEN__)</span>
            </div>
            <div style="display: inline-flex; align-items: center; gap: 5px; background: rgba(0, 0, 0, 0.7); padding: 4px 10px; border-radius: 12px; border: 1px solid #00BBF9;">
                <span style="width: 8px; height: 8px; background: #00BBF9; border-radius: 50%; box-shadow: 0 0 6px #00BBF9;"></span>
                <span style="color: #E2E8F0; font-weight: 600;">Additional (__ADDITIONAL_LEN__)</span>
            </div>
        </div>

        <div style="position: absolute; bottom: 10px; right: 16px; z-index: 10; color: #64748B; font-family: 'Segoe UI', sans-serif; font-size: 10px; pointer-events: none;">
            Drag to rotate in 3D • Scroll to zoom
        </div>

        <div id="three-skill-constellation-container" style="width: 100%; height: 100%;"></div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {
        const container = document.getElementById('three-skill-constellation-container');
        if (!container) return;

        const nodes = __NODES_JSON__;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(0, 0, 24);

        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const graphGroup = new THREE.Group();
        scene.add(graphGroup);

        function makeTextSprite(message, color) {
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 64;
            const ctx = canvas.getContext('2d');
            ctx.font = 'bold 20px "Segoe UI", Arial, sans-serif';
            ctx.fillStyle = color;
            ctx.textAlign = 'center';
            ctx.shadowColor = 'rgba(0,0,0,0.8)';
            ctx.shadowBlur = 4;
            ctx.fillText(message, 128, 40);

            const texture = new THREE.CanvasTexture(canvas);
            const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(4, 1, 1);
            return sprite;
        }

        nodes.forEach(function(node) {
            const hex = parseInt(node.color.replace("#", "0x"), 16);
            const geo = new THREE.SphereGeometry(node.size * 0.4, 16, 16);
            const mat = new THREE.MeshBasicMaterial({
                color: hex,
                wireframe: node.type.startsWith("core")
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(node.x, node.y, node.z);
            graphGroup.add(mesh);

            const sprite = makeTextSprite(node.label, node.color);
            sprite.position.set(node.x, node.y + (node.size * 0.5 + 0.6), node.z);
            graphGroup.add(sprite);

            if (node.type === "matched") {
                const lineGeo = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(-6, 0, 0),
                    new THREE.Vector3(node.x, node.y, node.z),
                    new THREE.Vector3(6, 0, 0)
                ]);
                const lineMat = new THREE.LineBasicMaterial({ color: 0x00F5D4, transparent: true, opacity: 0.35 });
                const line = new THREE.Line(lineGeo, lineMat);
                graphGroup.add(line);
            } else if (node.type === "missing") {
                const lineGeo = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(6, 0, 0),
                    new THREE.Vector3(node.x, node.y, node.z)
                ]);
                const lineMat = new THREE.LineBasicMaterial({ color: 0xFF007F, transparent: true, opacity: 0.25 });
                const line = new THREE.Line(lineGeo, lineMat);
                graphGroup.add(line);
            } else if (node.type === "additional") {
                const lineGeo = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(-6, 0, 0),
                    new THREE.Vector3(node.x, node.y, node.z)
                ]);
                const lineMat = new THREE.LineBasicMaterial({ color: 0x00BBF9, transparent: true, opacity: 0.25 });
                const line = new THREE.Line(lineGeo, lineMat);
                graphGroup.add(line);
            }
        });

        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };

        container.addEventListener('mousedown', function(e) {
            isDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener('mouseup', function() {
            isDragging = false;
        });

        container.addEventListener('mousemove', function(e) {
            if (!isDragging) return;
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;

            graphGroup.rotation.y += deltaX * 0.008;
            graphGroup.rotation.x += deltaY * 0.008;

            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        container.addEventListener('wheel', function(e) {
            camera.position.z += e.deltaY * 0.02;
            camera.position.z = Math.max(10, Math.min(45, camera.position.z));
            e.preventDefault();
        });

        function animate() {
            requestAnimationFrame(animate);
            if (!isDragging) {
                graphGroup.rotation.y += 0.003;
            }
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', function() {
            if (!container) return;
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    })();
    </script>
    """
    html = template.replace("__MATCHED_LEN__", str(len(matched_skills)))
    html = html.replace("__MISSING_LEN__", str(len(missing_skills)))
    html = html.replace("__ADDITIONAL_LEN__", str(len(additional_skills)))
    html = html.replace("__NODES_JSON__", nodes_json)
    return html


def plot_3d_category_vector_space(
    resume_text: str,
    jd_text: str,
    classifier_obj: Any
) -> go.Figure:
    """Generates an interactive Plotly 3D Semantic Scatter Plot comparing the Resume, JD, and Career Clusters."""
    from model import SAMPLE_RESUME_DATA

    texts = [clean_text(r[0]) for r in SAMPLE_RESUME_DATA]
    categories = [r[1] for r in SAMPLE_RESUME_DATA]

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    all_texts = texts + [cleaned_resume, cleaned_jd]

    vectorizer = classifier_obj.vectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    else:
        try:
            tfidf_matrix = vectorizer.transform(all_texts)
        except Exception:
            tfidf_matrix = TfidfVectorizer(max_features=1000).fit_transform(all_texts)

    svd = TruncatedSVD(n_components=3, random_state=42)
    coords_3d = svd.fit_transform(tfidf_matrix)

    n_samples = len(texts)
    df_plot = pd.DataFrame({
        "x": coords_3d[:n_samples, 0],
        "y": coords_3d[:n_samples, 1],
        "z": coords_3d[:n_samples, 2],
        "Category": categories,
        "Type": "Reference Benchmark"
    })

    fig = go.Figure()

    color_palette = [
        "#00F5D4", "#7B2CBF", "#00BBF9", "#FEE440", "#F15BB5",
        "#3A86FF", "#FF006E", "#8338EC", "#FB5607", "#06D6A0",
        "#118AB2", "#EF476F"
    ]

    for i, cat in enumerate(sorted(df_plot["Category"].unique())):
        cat_df = df_plot[df_plot["Category"] == cat]
        fig.add_trace(go.Scatter3d(
            x=cat_df["x"],
            y=cat_df["y"],
            z=cat_df["z"],
            mode="markers+text",
            name=cat,
            text=cat_df["Category"],
            textposition="top center",
            marker=dict(
                size=6,
                color=color_palette[i % len(color_palette)],
                opacity=0.85,
                line=dict(width=1, color="#FFFFFF")
            )
        ))

    fig.add_trace(go.Scatter3d(
        x=[coords_3d[n_samples, 0]],
        y=[coords_3d[n_samples, 1]],
        z=[coords_3d[n_samples, 2]],
        mode="markers+text",
        name="YOUR RESUME",
        text=["📍 YOUR RESUME"],
        textposition="top center",
        marker=dict(
            size=14,
            symbol="diamond",
            color="#00F5D4",
            opacity=1.0,
            line=dict(width=3, color="#FFFFFF")
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=[coords_3d[n_samples + 1, 0]],
        y=[coords_3d[n_samples + 1, 1]],
        z=[coords_3d[n_samples + 1, 2]],
        mode="markers+text",
        name="TARGET JOB DESCRIPTION",
        text=["🎯 TARGET JOB"],
        textposition="top center",
        marker=dict(
            size=14,
            symbol="diamond",
            color="#FF007F",
            opacity=1.0,
            line=dict(width=3, color="#FFFFFF")
        )
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10, 14, 26, 0.95)",
        plot_bgcolor="rgba(10, 14, 26, 0.95)",
        margin=dict(l=0, r=0, b=0, t=30),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#94A3B8")
        ),
        scene=dict(
            xaxis=dict(
                title="Semantic Latent X",
                backgroundcolor="rgba(15, 23, 42, 0.6)",
                gridcolor="rgba(255, 255, 255, 0.1)",
                showbackground=True,
                zerolinecolor="rgba(0, 245, 212, 0.4)"
            ),
            yaxis=dict(
                title="Skill Embedding Y",
                backgroundcolor="rgba(15, 23, 42, 0.6)",
                gridcolor="rgba(255, 255, 255, 0.1)",
                showbackground=True,
                zerolinecolor="rgba(0, 245, 212, 0.4)"
            ),
            zaxis=dict(
                title="Domain Density Z",
                backgroundcolor="rgba(15, 23, 42, 0.6)",
                gridcolor="rgba(255, 255, 255, 0.1)",
                showbackground=True,
                zerolinecolor="rgba(0, 245, 212, 0.4)"
            ),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)
            )
        )
    )

    return fig


def plot_3d_capability_mesh(category_breakdown: Dict[str, Any]) -> go.Figure:
    """Generates a 3D Bar / Mesh chart comparing Candidate Skills vs Required JD Skills across domains."""
    categories = list(category_breakdown.keys())[:7]
    if not categories:
        categories = ["Programming", "Frameworks", "AI/ML", "Databases", "DevOps", "QA", "Soft Skills"]

    matched_counts = [category_breakdown.get(c, {}).get("total_matched", 1) for c in categories]
    required_counts = [category_breakdown.get(c, {}).get("total_required", 2) for c in categories]
    coverage_scores = [category_breakdown.get(c, {}).get("coverage_percent", 50.0) for c in categories]

    x_grid = np.arange(len(categories))
    y_grid = np.array([0, 1, 2])
    X, Y = np.meshgrid(x_grid, y_grid)

    Z = np.zeros((3, len(categories)))
    for j in range(len(categories)):
        Z[0, j] = required_counts[j]
        Z[1, j] = matched_counts[j]
        Z[2, j] = coverage_scores[j] / 20.0

    fig = go.Figure(data=[
        go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale="Viridis",
            opacity=0.85,
            showscale=False
        )
    ])

    fig.update_layout(
        title="3D Capability Topology Across Domains",
        template="plotly_dark",
        paper_bgcolor="rgba(10, 14, 26, 0.95)",
        margin=dict(l=0, r=0, b=0, t=40),
        height=380,
        scene=dict(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(categories))),
                ticktext=[c.split()[0] for c in categories],
                title="Domain"
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["Required", "Matched", "Coverage"],
                title="Metric"
            ),
            zaxis=dict(title="Depth Level"),
            camera=dict(eye=dict(x=1.7, y=-1.5, z=1.3))
        )
    )

    return fig
