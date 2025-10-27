// 专业气液两相流流型动画系统 - 基于实际物理特征设计
// 流型动画系统 - 7种流型可视化

// 段塞流：
// 伪段塞流
// 分层波浪流
// 分层光滑流
// 泡沫段塞流
// 分层泡沫波浪流
// 泡沫环状流

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as TSL from "three/tsl";
import * as WEBGPU from "three/webgpu";

// 创建透明管道的函数
function createTransparentPipe(scene) {
  const geometry = new THREE.CylinderGeometry(
    pipeRadius,
    pipeRadius,
    pipeLength,
    32,
  ); // 改为横向圆柱
  const material = new THREE.MeshPhongMaterial({
    // color: "#4488ff",
    color: "#424242",
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
  });
  const cylinder = new THREE.Mesh(geometry, material);
  cylinder.rotation.z = Math.PI / 2; // 将圆柱体绕z轴旋转90度，使其沿x轴横向

  // 重要：设置渲染顺序，透明物体应该最后渲染
  cylinder.renderOrder = 1; // 设置渲染顺序，确保管道在液体之后渲染

  scene.add(cylinder);
  return cylinder;
}

function createWater() {
  const textureLoader = new THREE.TextureLoader();
  const iceDiffuse = textureLoader.load("assets/textures/water.png");
  iceDiffuse.wrapS = THREE.RepeatWrapping;
  iceDiffuse.wrapT = THREE.RepeatWrapping;
  iceDiffuse.colorSpace = THREE.NoColorSpace;

  const triplanarTexture = (...params) => TSL.triplanarTextures(...params);
  const iceColorNode = triplanarTexture(TSL.texture(iceDiffuse))
    .add(TSL.color(0x0066ff))
    .mul(0.8);

  // const geometry2 = new THREE.IcosahedronGeometry( 1, 3 );
  // const material = new THREE.MeshStandardNodeMaterial( { colorNode: iceColorNode } );

  const timer = TSL.time.mul(0.8);

  // *** 速度、方向 *** //

  // const floorUV = TSL.positionWorld.xzy;
  // const waterLayer0 = TSL.mx_worley_noise_float( floorUV.mul( 4 ).add( timer ) );
  // const waterLayer1 = TSL.mx_worley_noise_float( floorUV.mul( 2 ).add( timer ) );

  const waveDirection = TSL.vec2(-1.0, 0.5).normalize(); // 控制波浪传播方向
  const waveSpeed = 0.8; // 波浪速度

  // 应用方向到 UV
  const directionalUV = TSL.positionWorld.xz.add(
    TSL.vec2(waveDirection.x, waveDirection.y).mul(timer.mul(waveSpeed)),
  );

  const waterLayer0 = TSL.mx_worley_noise_float(directionalUV.mul(4));
  const waterLayer1 = TSL.mx_worley_noise_float(directionalUV.mul(2));
  // *** 速度、方向 *** //

  const waterIntensity = waterLayer0.mul(waterLayer1);
  // const waterColor = waterIntensity.mul( 1.4 ).mix( TSL.color( 0x0487e2 ), TSL.color( 0x74ccf4 ) );
  const waterColor = waterIntensity
    .mul(1.4)
    .mix(TSL.color("#0487e2"), TSL.color("#74ccf4"));

  // linearDepth() returns the linear depth of the mesh
  const depth = TSL.linearDepth();
  const depthWater = TSL.viewportLinearDepth.sub(depth);
  const depthEffect = depthWater.remapClamp(-0.002, 0.04);

  const wavyHeight = 1.5; // .1  0.5  1.5
  const refractionUV = TSL.screenUV.add(
    TSL.vec2(0, waterIntensity.mul(wavyHeight)),
  );

  // linearDepth( viewportDepthTexture( uv ) ) return the linear depth of the scene
  const depthTestForRefraction = TSL.linearDepth(
    TSL.viewportDepthTexture(refractionUV),
  ).sub(depth);

  const depthRefraction = depthTestForRefraction.remapClamp(0, 0.1);

  const finalUV = depthTestForRefraction
    .lessThan(0)
    .select(TSL.screenUV, refractionUV);

  const viewportTexture = TSL.viewportSharedTexture(finalUV);

  const waterMaterial = new WEBGPU.MeshBasicNodeMaterial({
    colorNode: waterColor,
    transparent: true,
    backdropAlphaNode: depthRefraction.oneMinus(),
    backdropNode: depthEffect.mix(
      TSL.viewportSharedTexture(),
      viewportTexture.mul(depthRefraction.mix(1, waterColor)),
    ),
    opacity: 0.5,
  });

  // 可控制的参数
  const waterParams = {
    direction: [1.0, 0.5], // 波浪方向 [x, y]
    speed: 0.8, // 波浪速度
    scale: 4.0, // 波浪尺度
    intensity: 1.4, // 波浪强度
    waveHeight: 10, // 波浪高度
  };

  // const water = new WEBGPU.Mesh( new WEBGPU.BoxGeometry( 50, .001, 50 ), waterMaterial );
  const water = new WEBGPU.Mesh(
    new WEBGPU.BoxGeometry(pipeLength, 0.001, 0.3),
    waterMaterial,
  );
  water.position.set(0, 0, 0);

  return { water: water, waterMaterial: waterMaterial };
}
// 创建段塞流
function createSlugFlow(scene) {
  const liquidGroup = new THREE.Group();

  const segmentCount = 7;
  const segmentLength = 0.3;
  const gapLength = 0.5;
  const radius = 0.15;

  // const geometry = new THREE.CylinderGeometry(radius, radius, segmentLength, 32);
  // 创建更真实的液体几何体 - 使用球体端盖的胶囊形状
  const geometry = new THREE.CapsuleGeometry(radius, segmentLength, 16, 16);

  const material = new THREE.MeshPhongMaterial({
    // color: '#0000ff',
    color: "#4488ff",
    transparent: true,
    opacity: 0.5,

    specular: "#ffffff", // 高光颜色
    shininess: 100, // 光泽度
    reflectivity: 0.5, // 反射率
  });

  for (let i = 0; i < segmentCount; i++) {
    const cylinder = new THREE.Mesh(geometry, waterMaterial);
    cylinder.rotation.z = Math.PI / 2;

    // 重要：液体应该在管道之前渲染
    cylinder.renderOrder = 0;

    // 初始位置：每个段塞流段的位置, 从管道左侧开始
    cylinder.position.set(
      pipeLeftBoundary + segmentLength / 2 + i * (segmentLength + gapLength),
      0,
      0,
    );

    // 为每个液体段添加独特的纹理偏移
    cylinder.userData = {
      textureOffset: Math.random() * 100,
      wavePhase: Math.random() * Math.PI * 2,
      bubbleIntensity: 0.5 + Math.random() * 0.5,
      originalScale: new THREE.Vector3().copy(cylinder.scale),
    };

    liquidGroup.add(cylinder);
  }

  scene.add(liquidGroup);

  // 更新液体流动 - 精确控制版本
  function updateLiquidFlow(liquidGroup) {
    const flowSpeed = 0.15;
    const currentTime = Date.now() * 0.001;

    liquidGroup.children.forEach((segment, index) => {
      let currentX = segment.position.x;
      // 向右移动
      currentX += flowSpeed * 0.016; // 假设60fps，每帧移动距离

      // 检查是否完全流出管道右侧
      // const segmentRightEdge = currentX - segmentLength / 2 - radius;
      const segmentRightEdge = currentX - segmentLength / 2 - radius * 2;
      if (segmentRightEdge > pipeRightBoundary) {
        // 找到最左侧的液体段位置
        let leftmostX = pipeRightBoundary;

        liquidGroup.children.forEach((s, i) => {
          if (s.position.x < leftmostX) leftmostX = s.position.x;
        });

        // 回到所有液体段的最左侧
        currentX = leftmostX - (segmentLength + gapLength);
      }

      // 设置位置
      segment.position.x = currentX;

      const userData = segment.userData;
      const material = segment.material;
      // 旋转效果 - 模拟液体内部流动
      // segment.rotation.x = Math.sin(currentTime * 1.2 + index * 0.3) * 0.08;
      // 气泡效果 - 动态改变粗糙度
      // const bubbleEffect = Math.sin(currentTime * 3 + userData.wavePhase) * 0.05;
      // material.roughness = 5.15 + bubbleEffect * userData.bubbleIntensity;
      // 表面张力效果 - 轻微的形状变化
      const surfaceTension = Math.sin(currentTime * 2.5 + index) * 0.02;
      segment.scale.y = userData.originalScale.y + surfaceTension;
      segment.scale.z = userData.originalScale.z - surfaceTension * 0.5;
    });
  }
  return { slugGroup: liquidGroup, updateSlugFlow: updateLiquidFlow };
}

// 创建半个圆柱体的函数
function createHalfCylinder(options = {}) {
  const {
    radiusTop = 0.15,
    radiusBottom = 0.15,
    height = 3,
    radialSegments = 32,
    position = { x: 0, y: 0, z: 0 },
    rotation = { x: 0, y: 0, z: -Math.PI / 2 },
    halfType = "vertical", // 'vertical' 或 'horizontal'
    MeshPhongMaterial = {
      color: "#4488ff",
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide, // 双面渲染，确保半圆柱体看起来完整
    },
  } = options;

  // 创建圆柱体几何体，但只使用一半的面
  const geometry = new THREE.CylinderGeometry(
    radiusTop,
    radiusBottom,
    height,
    radialSegments,
  );

  // 获取几何体的顶点和面数据
  const positionAttribute = geometry.getAttribute("position");
  const indices = geometry.getIndex();

  // 根据半圆柱体类型过滤顶点
  const verticesToKeep = new Set();

  if (halfType === "vertical") {
    // 垂直切割：保留y轴正方向的一半
    for (let i = 0; i < positionAttribute.count; i++) {
      const x = positionAttribute.getX(i);
      if (x >= 0) {
        // 保留x轴正方向的一半
        verticesToKeep.add(i);
      }
    }
  } else {
    // 水平切割：保留上半部分
    for (let i = 0; i < positionAttribute.count; i++) {
      const y = positionAttribute.getY(i);
      if (y >= 0) {
        // 保留y轴正方向的一半
        verticesToKeep.add(i);
      }
    }
  }

  // 创建新的索引数组，只保留需要的面
  const newIndices = [];
  if (indices) {
    for (let i = 0; i < indices.count; i += 3) {
      const a = indices.getX(i);
      const b = indices.getX(i + 1);
      const c = indices.getX(i + 2);

      // 只有当三角形的所有顶点都在保留范围内时才保留这个面
      if (
        verticesToKeep.has(a) &&
        verticesToKeep.has(b) &&
        verticesToKeep.has(c)
      ) {
        newIndices.push(a, b, c);
      }
    }
  }

  // 创建新的几何体
  const halfGeometry = new THREE.BufferGeometry();

  // 复制顶点属性
  const newPositions = [];
  const vertexMap = new Map();
  let newIndex = 0;

  const uvAttribute = geometry.getAttribute("uv");
  const newUVs = []; // uv 坐标数组, 贴图需要

  for (let i = 0; i < positionAttribute.count; i++) {
    if (verticesToKeep.has(i)) {
      newPositions.push(
        positionAttribute.getX(i),
        positionAttribute.getY(i),
        positionAttribute.getZ(i),
      );
      // 复制UV坐标（如果存在）
      if (uvAttribute) {
        newUVs.push(uvAttribute.getX(i), uvAttribute.getY(i));
      }
      vertexMap.set(i, newIndex);
      newIndex++;
    }
  }

  // 重新映射索引
  const remappedIndices = newIndices.map((oldIndex) => vertexMap.get(oldIndex));

  // 设置UV坐标（如果复制了UV数据）
  if (newUVs.length > 0) {
    // console.log("newUVs", newUVs);
    halfGeometry.setAttribute(
      "uv",
      new THREE.Float32BufferAttribute(newUVs, 2),
    );
  } else {
    // 如果没有UV坐标，创建简单的UV映射
    halfGeometry.computeBoundingBox();
    const bbox = halfGeometry.boundingBox;
    const size = new THREE.Vector3();
    bbox.getSize(size);

    const uvs = [];
    for (let i = 0; i < newPositions.length / 3; i++) {
      const x = newPositions[i * 3];
      const y = newPositions[i * 3 + 1];
      const z = newPositions[i * 3 + 2];

      // 简单的UV映射
      const u = (x - bbox.min.x) / size.x;
      const v = (z - bbox.min.z) / size.z;
      uvs.push(u, v);
    }
    halfGeometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  }

  halfGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(newPositions, 3),
  );
  halfGeometry.setIndex(remappedIndices);
  halfGeometry.computeVertexNormals();
  halfGeometry.userData = {
    name: "halfCylinder",
  };

  const material = new THREE.MeshPhongMaterial(MeshPhongMaterial);
  // const material = new THREE.MeshPhongNodeMaterial(MeshPhongMaterial);
  // const material = new THREE.MeshBasicNodeMaterial(MeshPhongMaterial);

  const halfCylinder = new THREE.Mesh(halfGeometry, material);
  halfCylinder.position.set(position.x, position.y, position.z);
  halfCylinder.rotation.set(rotation.x, rotation.y, rotation.z);

  halfCylinder.renderOrder = 0;

  return halfCylinder;
}

// 创建伪段塞流
function createPseudoSlugFlow(scene) {
  const pseudoSlugGroup = new THREE.Group();

  const segmentCount = 7;
  const segmentLength = 0.3;
  const gapLength = 0.5;
  const radius = 0.15;

  // const geometry = new THREE.CylinderGeometry(radius, radius, segmentLength, 32);
  // 创建更真实的液体几何体 - 使用球体端盖的胶囊形状
  const geometry = new THREE.CapsuleGeometry(radius, segmentLength, 16, 16);

  const material = new THREE.MeshPhongMaterial({
    // color: '#0000ff',
    color: "#4488ff",
    transparent: true,
    opacity: 0.5,

    specular: "#ffffff", // 高光颜色
    shininess: 100, // 光泽度
    reflectivity: 0.5, // 反射率
  });

  for (let i = 0; i < segmentCount; i++) {
    // const cylinder = new THREE.Mesh(geometry, material);
    const cylinder = new THREE.Mesh(geometry, waterMaterial);
    cylinder.rotation.z = Math.PI / 2;

    // 重要：液体应该在管道之前渲染
    cylinder.renderOrder = 0;

    // 初始位置：每个段塞流段的位置, 从管道左侧开始
    cylinder.position.set(
      pipeLeftBoundary + segmentLength / 2 + i * (segmentLength + gapLength),
      0,
      0,
    );

    // 为每个液体段添加独特的纹理偏移
    cylinder.userData = {
      textureOffset: Math.random() * 100,
      wavePhase: Math.random() * Math.PI * 2,
      bubbleIntensity: 0.5 + Math.random() * 0.5,
      originalScale: new THREE.Vector3().copy(cylinder.scale),
    };

    pseudoSlugGroup.add(cylinder);
  }

  // 创建半个圆柱体 - 竖直切割
  const halfCylinderHorizontal = createHalfCylinder({ height: pipeLength });
  pseudoSlugGroup.add(halfCylinderHorizontal);

  scene.add(pseudoSlugGroup);

  // 动画
  function updatePseudoSlugFlow(liquidGroup) {
    const flowSpeed = 0.15;

    liquidGroup.children.forEach((segment, index) => {
      let currentX = segment.position.x;
      // 向右移动
      currentX += flowSpeed * 0.016; // 假设60fps，每帧移动距离

      if (segment.geometry.userData.name !== "halfCylinder") {
        // 检查是否完全流出管道右侧
        // const segmentRightEdge = currentX - segmentLength / 2 - radius;
        const segmentRightEdge = currentX - segmentLength / 2 - radius * 2;
        if (segmentRightEdge > pipeRightBoundary) {
          // 找到最左侧的液体段位置
          let leftmostX = pipeRightBoundary;

          liquidGroup.children.forEach((s, i) => {
            if (s.geometry.userData.name !== "halfCylinder")
              if (s.position.x < leftmostX) leftmostX = s.position.x;
          });

          // 回到所有液体段的最左侧
          currentX = leftmostX - (segmentLength + gapLength);
        }

        // 设置位置
        segment.position.x = currentX;
      }
    });
  }

  return {
    pseudoSlugGroup: pseudoSlugGroup,
    updatePseudoSlugFlow: updatePseudoSlugFlow,
  };
}

// 创建分层波浪流
function createStratifiedWavyFlow(scene) {
  const stratifiedWavyGroup = new THREE.Group();

  // 创建波浪形状的液体段
  const waveSegmentCount = 7; // 增加段数以获得更平滑的波浪
  const waveAmplitude = 0.05; // 波浪幅度
  const waveFrequency = 2.0; // 波浪频率
  const segmentLength = 0.2; // 每段长度
  const radius = 0.1; // 液体半径

  // 创建波浪材质
  const waveMaterial = new THREE.MeshPhongMaterial({
    color: "#4488ff",
    transparent: true,
    opacity: 0.7,
    specular: "#ffffff",
    shininess: 80,
    reflectivity: 0.6,
  });
  const geometry = new THREE.CapsuleGeometry(radius, segmentLength, 8, 8);

  // 创建波浪形状的液体段
  for (let i = 0; i < waveSegmentCount; i++) {
    // 使用胶囊几何体创建波浪段
    // const waveSegment = new THREE.Mesh(geometry, waveMaterial);
    const waveSegment = new THREE.Mesh(geometry, waterMaterial);

    // 设置初始位置和旋转
    waveSegment.rotation.z = Math.PI / 2; // 横向放置
    waveSegment.position.x = pipeLeftBoundary + i * segmentLength + i * 0.5;

    // 为每个波浪段添加独特的波浪参数
    waveSegment.userData = {
      wavePhase: (i / waveSegmentCount) * Math.PI * 2, // 相位偏移
      waveAmplitude: waveAmplitude,
      waveFrequency: waveFrequency,
      originalY: 0,
      isWaveSegment: true,
    };

    stratifiedWavyGroup.add(waveSegment);
  }

  const halfCylinderHorizontal = createHalfCylinder({ height: pipeLength });

  stratifiedWavyGroup.add(halfCylinderHorizontal);

  scene.add(stratifiedWavyGroup);

  function updateStratifiedWavyFlow(liquidGroup) {
    const flowSpeed = 0.15;
    const currentTime = Date.now() * 0.001;

    liquidGroup.children.forEach((segment, index) => {
      let currentX = segment.position.x;
      // 向右移动
      currentX += flowSpeed * 0.016; // 假设60fps，每帧移动距离

      if (segment.userData.isWaveSegment) {
        // 检查是否流出管道
        const segmentRightEdge = currentX - segmentLength / 2 - radius / 2;
        if (segmentRightEdge > pipeRightBoundary) {
          // 重置到管道左侧
          currentX = pipeLeftBoundary - segmentLength / 2;
        }

        // 设置X位置
        segment.position.x = currentX;

        // 计算波浪效果
        const wavePhase =
          segment.userData.wavePhase +
          currentTime * segment.userData.waveFrequency;
        const waveOffset = Math.sin(wavePhase) * segment.userData.waveAmplitude;

        // 应用波浪偏移到Y轴
        segment.position.y = segment.userData.originalY + waveOffset;

        // 添加轻微的旋转效果模拟波浪
        segment.rotation.y = Math.sin(wavePhase * 0.5) * 0.1;
      }
    });
  }

  return {
    stratifiedWavyGroup: stratifiedWavyGroup,
    updateStratifiedWavyFlow: updateStratifiedWavyFlow,
  };
}

// 创建分层平滑流
function createStratifiedSmoothFlow(scene) {
  const stratifiedSmoothGroup = new THREE.Group();

  const halfCylinderHorizontal = createHalfCylinder({
    height: pipeLength,
    MeshPhongMaterial: {
      color: "#4488ff",
      opacity: 0.5,
      transparent: true,
      side: THREE.DoubleSide, // 双面渲染，确保半圆柱体看起来完整
    },
  });
  const halfCylinderHorizontal2 = createHalfCylinder({
    position: { x: -3, y: 0, z: 0 },
  });

  // stratifiedSmoothGroup.add(halfCylinderHorizontal, halfCylinderHorizontal2);
  stratifiedSmoothGroup.add(halfCylinderHorizontal);

  scene.add(stratifiedSmoothGroup);

  function updateStratifiedSmoothFlow(liquidGroup) {
    // const flowSpeed = 0.15;
    // const currentTime = Date.now() * 0.001;
    // liquidGroup.children.forEach((segment, index) => {
    //     let currentX = segment.position.x;
    //     // 向右移动
    //     currentX += flowSpeed * 0.016;  // 假设60fps，每帧移动距离
    //     if ((currentX-pipeLength/2) > pipeRightBoundary){
    //         currentX = pipeLeftBoundary*2;
    //     }
    //     segment.position.x = currentX;
    // });
  }

  return {
    stratifiedSmoothGroup: stratifiedSmoothGroup,
    updateStratifiedSmoothFlow: updateStratifiedSmoothFlow,
  };
}

/**
 * 通用 WebGPU 泡沫粒子系统
 * 支持全圆液体或下半圆液体
 *
 * @param {THREE.Scene} scene
 * @param {Object} options
 * @param {number} [options.pipeLength=5] 管道长度
 * @param {number} [options.liquidRadius=0.15] 液体半径
 * @param {boolean} [options.half=false] 是否为下半圆液体
 * @param {number} [options.count=800] 泡沫数量
 * @returns {Object} { updateFoam(dt), flowSpeed, points }
 */
function createFoam(scene, options = {}) {
  const COUNT = options.count ?? 800;
  const pipeLength = options.pipeLength ?? 5;
  const liquidRadius = options.liquidRadius ?? 0.15;
  const isHalf = options.half ?? false; // true => 下半圆流
  const flowSpeedDefault = 1.0;

  // === 1. 准备几何 ===
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(COUNT * 3);
  const velocities = new Float32Array(COUNT * 3);
  const sizes = new Float32Array(COUNT);
  const uvs = new Float32Array(COUNT * 2);

  // === 2. 初始化泡沫粒子 ===
  for (let i = 0; i < COUNT; i++) {
    const ix = i * 3;

    // X：沿管道方向分布（不靠边）
    positions[ix] = Math.random() * (pipeLength * 0.9) - pipeLength / 2;

    // Y/Z：生成在液体圆截面内（或下半圆）
    const r = Math.sqrt(Math.random()) * liquidRadius;
    const theta = isHalf
      ? Math.random() * Math.PI // 下半圆
      : Math.random() * Math.PI * 2; // 全圆

    let y = r * Math.cos(theta);
    let z = r * Math.sin(theta);

    if (isHalf) y = -Math.abs(y); // 让下半圆部分都在下方

    positions[ix + 1] = y;
    positions[ix + 2] = z;

    // 速度
    velocities[ix] = 0.03 + Math.random() * 0.03;
    velocities[ix + 1] = Math.random() * 0.01 - 0.005;
    velocities[ix + 2] = Math.random() * 0.01 - 0.005;

    sizes[i] = 3 + Math.random() * 8;
    uvs[i * 2] = Math.random();
    uvs[i * 2 + 1] = Math.random();
  }

  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));

  // === 3. 泡沫贴图 ===
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(64, 64, 5, 64, 64, 60);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.4)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 128, 128);
  const bubbleTex = new THREE.CanvasTexture(canvas);

  // === 4. 材质 ===
  const mat = new WEBGPU.PointsNodeMaterial({
    transparent: true,
    depthWrite: false,
    sizeAttenuation: true,
    blending: THREE.AdditiveBlending,
    color: new THREE.Color(0xffffff),
    map: bubbleTex,
  });

  // === 5. 创建对象 ===
  const points = new THREE.Points(geo, mat);
  scene.add(points);

  // === 6. 更新逻辑 ===
  let flowSpeed = flowSpeedDefault;

  function randomYZ() {
    const r = Math.sqrt(Math.random()) * liquidRadius;
    const theta = isHalf
      ? Math.random() * Math.PI
      : Math.random() * Math.PI * 2;
    let y = r * Math.cos(theta);
    let z = r * Math.sin(theta);
    if (isHalf) y = -Math.abs(y);
    return { y, z };
  }

  function updateFoam(dt = 0.016) {
    for (let i = 0; i < COUNT; i++) {
      const ix = i * 3;
      positions[ix] += velocities[ix] * dt * 60 * flowSpeed;

      // 循环流动
      if (positions[ix] > pipeLength / 2) {
        positions[ix] = -pipeLength / 2;
        const { y, z } = randomYZ();
        positions[ix + 1] = y;
        positions[ix + 2] = z;
      }
    }
    geo.attributes.position.needsUpdate = true;
  }

  return {
    updateFoam,
    get flowSpeed() {
      return flowSpeed;
    },
    set flowSpeed(v) {
      flowSpeed = Math.max(0, v);
    },
    points,
  };
}

// 创建泡沫段塞流
function createFoamySlugFlow(scene) {}

// 创建分层泡沫波浪流
function createStratifiedFoamyWavyFlow(scene) {}

// 创建泡沫环状流
function createFoamyAnnularFlow(scene) {
  const foamyAnnular = new THREE.Group();

  const radius = 0.15;
  const geometry = new THREE.CylinderGeometry(radius, radius, pipeLength, 32); // 改为横向圆柱
  const material = new THREE.MeshPhongMaterial({
    color: "#4488ff",
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
  });

  const cylinder = new THREE.Mesh(geometry, waterMaterial);
  cylinder.rotation.z = Math.PI / 2; // 将圆柱体绕z轴旋转90度，使其沿x轴横向
  cylinder.renderOrder = 0;

  foamyAnnular.add(cylinder);
  scene.add(foamyAnnular);

  return { foamyAnnular: foamyAnnular };
}

// 管道参数
const pipeRadius = 0.2; // 管道半径
// const pipeLength = 3; // 管道长度
const pipeLength = 5; // 管道长度
const pipeLeftBoundary = -pipeLength / 2; // 管道左边界：-1
const pipeRightBoundary = pipeLength / 2; // 管道右边界：1

const { water, waterMaterial } = createWater();

function main() {
  // 获取canvas画布元素
  const canvas = document.querySelector(".flow-animation-canvas");
  // 创建场景
  const scene = new THREE.Scene();

  // 创建渲染器
  // const renderer = new THREE.WebGLRenderer({
  //     antialias:true, canvas:canvas,
  //     alpha: true // 允许透明背景
  // });

  const renderer = new WEBGPU.WebGPURenderer({
    antialias: true,
    canvas: canvas,
    alpha: true, // 允许透明背景
  });

  renderer.setSize(canvas.clientWidth, canvas.clientHeight); // 渲染器的大小设置为画布的大小
  renderer.setPixelRatio(window.devicePixelRatio); // 设置像素比以适应高分辨率屏幕
  // renderer.sortObjects = true; // 重要：启用透明度排序

  // 创建相机
  const camera = new THREE.PerspectiveCamera(
    45,
    canvas.clientWidth / canvas.clientHeight,
    0.1,
    500,
  );
  camera.position.set(0.1, 0.8, 1);
  camera.lookAt(0, 0, 0);

  // 创建轨道控制器
  //   const controls = new OrbitControls(camera, renderer.domElement);
  //   controls.target.set(0, 0, 0);
  //   controls.update();

  // 显示坐标轴
  //   const axesHelper = new THREE.AxesHelper();
  //   scene.add(axesHelper);

  // 添加环境光
  const ambientLight = new THREE.AmbientLight(0x909090, 10);
  scene.add(ambientLight);
  // 添加定向光(平行光)
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
  directionalLight.position.set(5, 5, 5);
  // scene.add(directionalLight);
  const pointLight = new THREE.PointLight(0xffffd0, 0.5); // 点光源
  pointLight.position.set(0, 2, 2);
  scene.add(pointLight);

  // 透明管道
  const transparentPipe = createTransparentPipe(scene);
  transparentPipe.visible = false;

  // 段塞流
  const { slugGroup, updateSlugFlow } = createSlugFlow(scene);
  slugGroup.visible = false; // 隐藏

  // 伪段塞流
  const { pseudoSlugGroup, updatePseudoSlugFlow } = createPseudoSlugFlow(scene);
  pseudoSlugGroup.visible = false; // 隐藏

  // 分层波浪流
  const { stratifiedWavyGroup, updateStratifiedWavyFlow } =
    createStratifiedWavyFlow(scene);
  stratifiedWavyGroup.visible = false; // 隐藏

  // 分层平滑流
  const { stratifiedSmoothGroup, updateStratifiedSmoothFlow } =
    createStratifiedSmoothFlow(scene);
  stratifiedSmoothGroup.visible = false; // 隐藏

  // 泡沫段塞流
  createFoamySlugFlow(scene);

  // 分层泡沫波浪流
  const foam_half = createFoam(scene, { half: true });
  foam_half.flowSpeed = 0.2;
  foam_half.points.visible = false;

  createStratifiedFoamyWavyFlow(scene);

  // 泡沫环状流
  const foam = createFoam(scene);
  foam.flowSpeed = 0.1;
  foam.points.visible = false;

  const { foamyAnnular } = createFoamyAnnularFlow(scene);
  foamyAnnular.visible = false;

  var flowType = "slug"; // 'slug' 'pseudoSlug' 'stratifiedWavy' 'stratifiedSmooth' 'foamySlug' 'stratifiedFoamyWavy' 'foamyAnnular'
  flowType = "pseudoSlug";
  flowType = "stratifiedWavy";
  // flowType = 2;
  // flowType = 'stratifiedSmooth';
  // flowType = 'foamySlug';
  // flowType = 'stratifiedFoamyWavy';
  // flowType = 'foamyAnnular';

  scene.add(water);
  water.visible = false;

  // 动画循环
  function animate() {
    flowType = flow_result.pre_label;
    requestAnimationFrame(animate);

    if (flowType != undefined) {
      transparentPipe.visible = true; // 显示管道
    }

    // 液体流动动画
    switch (true) {
      case ["slug", 0].includes(flowType):
        pseudoSlugGroup.visible = false;
        stratifiedSmoothGroup.visible = false; // 隐藏
        stratifiedWavyGroup.visible = false;
        foamyAnnular.visible = false;
        foam.points.visible = false;
        foam_half.points.visible = false;

        slugGroup.visible = true; // 显示
        water.visible = false;
        updateSlugFlow(slugGroup);
        break;
      case ["pseudoSlug", 1].includes(flowType):
        slugGroup.visible = false;
        stratifiedSmoothGroup.visible = false;
        stratifiedWavyGroup.visible = false;
        foamyAnnular.visible = false;
        foam.points.visible = false;
        foam_half.points.visible = false;

        pseudoSlugGroup.visible = true;
        water.visible = true;
        updatePseudoSlugFlow(pseudoSlugGroup);
        break;
      case ["stratifiedWavy", 2].includes(flowType):
        slugGroup.visible = false;
        pseudoSlugGroup.visible = false;
        stratifiedSmoothGroup.visible = false;
        foamyAnnular.visible = false;
        foam.points.visible = false;
        foam_half.points.visible = false;

        stratifiedWavyGroup.visible = true;
        water.visible = true;
        updateStratifiedWavyFlow(stratifiedWavyGroup);
        break;
      case ["stratifiedSmooth", 3].includes(flowType):
        slugGroup.visible = false;
        pseudoSlugGroup.visible = false;
        stratifiedWavyGroup.visible = false;
        foamyAnnular.visible = false;
        foam.points.visible = false;
        foam_half.points.visible = false;

        stratifiedSmoothGroup.visible = true;
        water.visible = true;
        updateStratifiedSmoothFlow(stratifiedSmoothGroup);
        break;
      case ["foamySlug", 4].includes(flowType):
        slugGroup.visible = false;
        pseudoSlugGroup.visible = false;
        stratifiedWavyGroup.visible = false;
        foamyAnnular.visible = false;
        stratifiedSmoothGroup.visible = false;
        water.visible = false;
        foam_half.points.visible = false;

        foam.points.visible = true;
        foam.updateFoam();

        slugGroup.visible = true; // 显示
        updateSlugFlow(slugGroup);
        break;
      case ["stratifiedFoamyWavy", 5].includes(flowType):
        slugGroup.visible = false;
        pseudoSlugGroup.visible = false;
        stratifiedSmoothGroup.visible = false;
        foamyAnnular.visible = false;
        foam.points.visible = false;

        stratifiedWavyGroup.visible = true;
        water.visible = true;
        updateStratifiedWavyFlow(stratifiedWavyGroup);

        foam_half.points.visible = true;
        foam_half.updateFoam();
        break;
      case ["foamyAnnular", 6].includes(flowType):
        slugGroup.visible = false;
        pseudoSlugGroup.visible = false;
        stratifiedWavyGroup.visible = false;
        stratifiedSmoothGroup.visible = false;
        water.visible = false;

        foamyAnnular.visible = true;
        foam.points.visible = true; // 泡沫
        foam.updateFoam();
        break;
      default:
        break;
    }

    renderer.renderAsync(scene, camera);
  }

  animate(); // 启动动画循环
}

main();
