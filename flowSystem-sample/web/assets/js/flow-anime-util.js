import * as THREE from "three";

// 平滑水流效果工具函数
// 创建没有波浪的平滑水流效果

/**
 * 创建平滑水流效果
 * @param {number} length - 水流长度
 * @param {number} width - 水流宽度
 * @param {Object} options - 配置选项
 * @returns {Object} 包含水流对象和更新函数的对象
 */
function createSmoothWaterFlow(length = 3, width = 0.3, options = {}) {
    const {
        color = "#4488ff",
        opacity = 0.6,
        flowSpeed = 1.0,
        enableTextureFlow = true
    } = options;

    // 创建水面几何体
    const geometry = new THREE.PlaneGeometry(length, width, 32, 16);
    
    let waterMaterial;
    let updateFunction;

    if (enableTextureFlow) {
        // 使用着色器材质实现纹理流动效果
        waterMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                flowSpeed: { value: flowSpeed },
                baseColor: { value: new THREE.Color(color) },
                opacity: { value: opacity },
                resolution: { value: new THREE.Vector2(length, width) }
            },
            vertexShader: `
                uniform float time;
                uniform float flowSpeed;
                uniform vec2 resolution;
                
                varying vec2 vUv;
                varying vec3 vPosition;
                
                void main() {
                    vUv = uv;
                    vPosition = position;
                    
                    // 平滑的顶点位移 - 非常轻微，几乎不可见
                    vec3 newPosition = position;
                    newPosition.y += sin(position.x * 2.0 + time * flowSpeed) * 0.001;
                    
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform float flowSpeed;
                uniform vec3 baseColor;
                uniform float opacity;
                uniform vec2 resolution;
                
                varying vec2 vUv;
                varying vec3 vPosition;
                
                // 生成平滑的流动纹理
                float smoothFlowPattern(vec2 uv, float time) {
                    // 创建平滑的流动条纹
                    float flow = sin(uv.x * 8.0 + time * flowSpeed * 2.0) * 0.3 + 0.7;
                    // 添加轻微的横向变化
                    float variation = sin(uv.y * 4.0 + time * 0.5) * 0.1 + 0.9;
                    return flow * variation;
                }
                
                // 生成颜色渐变
                vec3 colorGradient(float intensity) {
                    vec3 lightBlue = vec3(0.04, 0.53, 0.89); // #0487e2
                    vec3 lightCyan = vec3(0.45, 0.80, 0.96); // #74ccf4
                    return mix(lightBlue, lightCyan, intensity);
                }
                
                void main() {
                    // 基于位置生成流动强度
                    float flowIntensity = smoothFlowPattern(vUv, time);
                    
                    // 应用颜色渐变
                    vec3 waterColor = colorGradient(flowIntensity);
                    
                    // 透明度随流动轻微变化
                    float alpha = opacity * (0.9 + flowIntensity * 0.2);
                    
                    // 添加轻微的光照效果
                    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
                    vec3 normal = vec3(0.0, 1.0, 0.0);
                    float diffuse = max(dot(normal, lightDir), 0.0) * 0.3 + 0.7;
                    
                    vec3 finalColor = waterColor * diffuse;
                    
                    gl_FragColor = vec4(finalColor, alpha);
                }
            `,
            transparent: true,
            side: THREE.DoubleSide
        });

        updateFunction = function() {
            waterMaterial.uniforms.time.value = performance.now() * 0.001;
        };
    } else {
        // 使用内置材质实现简单版本
        waterMaterial = new THREE.MeshPhongMaterial({
            color: color,
            transparent: true,
            opacity: opacity,
            specular: "#ffffff",
            shininess: 80
        });

        // 创建Canvas纹理实现流动效果
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');
        
        let textureTime = 0;
        
        function createSmoothTexture(time) {
            ctx.fillStyle = color;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 绘制平滑的流动条纹
            const gradientCount = 8;
            const stripeWidth = canvas.width / gradientCount;
            
            for (let i = 0; i < gradientCount; i++) {
                const offset = (time * 100 + i * stripeWidth) % canvas.width;
                const gradient = ctx.createLinearGradient(offset, 0, offset + stripeWidth, 0);
                
                gradient.addColorStop(0, color);
                gradient.addColorStop(0.3, '#74ccf4');
                gradient.addColorStop(0.7, '#74ccf4');
                gradient.addColorStop(1, color);
                
                ctx.fillStyle = gradient;
                ctx.fillRect(offset, 0, stripeWidth, canvas.height);
            }
            
            return new THREE.CanvasTexture(canvas);
        }
        
        waterMaterial.map = createSmoothTexture(0);
        waterMaterial.needsUpdate = true;
        
        updateFunction = function() {
            textureTime += 0.01;
            waterMaterial.map = createSmoothTexture(textureTime);
            waterMaterial.needsUpdate = true;
        };
    }

    // 创建水面网格
    const water = new THREE.Mesh(geometry, waterMaterial);
    water.rotation.x = -Math.PI / 2; // 水平放置
    
    return {
        water: water,
        material: waterMaterial,
        update: updateFunction
    };
}

/**
 * 创建管道内的平滑水流效果
 * @param {THREE.Scene} scene - 场景对象
 * @param {Object} options - 配置选项
 * @returns {Object} 包含更新函数的对象
 */
export function createPipeSmoothWaterFlow(scene, options = {}) {
    const {
        pipeLength = 3,
        pipeWidth = 0.3,
        flowSpeed = 1.0,
        enableBubbles = false
    } = options;
    
    const waterFlow = createSmoothWaterFlow(pipeLength, pipeWidth, {
        color: "#4488ff",
        opacity: 0.6,
        flowSpeed: flowSpeed,
        enableTextureFlow: true
    });
    
    waterFlow.water.position.set(0, 0.001, 0);
    scene.add(waterFlow.water);
    
    // 可选：添加气泡效果
    let bubbles = null;
    if (enableBubbles) {
        bubbles = createBubbleEffect(scene, pipeLength, pipeWidth, 0.0003);
    }
    
    return {
        update: function() {
            waterFlow.update();
            if (bubbles && bubbles.update) {
                bubbles.update();
            }
        },
        waterObject: {
            water: waterFlow.water,
            material: waterFlow.material,
            bubbles: bubbles
        },
        set visible(visible) {
            waterFlow.water.visible = visible;
            if (bubbles) {
                bubbles.bubbleGroup.visible = visible;
            }
        }
    };
}

/**
 * 创建气泡效果（可选）
 * @param {THREE.Scene} scene - 场景对象
 * @param {number} length - 管道长度
 * @param {number} width - 管道宽度
 * @returns {Object} 气泡效果对象
 */
function createBubbleEffect(scene, length, width, speed) {
    const bubbleGroup = new THREE.Group();
    const bubbleCount = 10;
    const bubbles = [];
    
    for (let i = 0; i < bubbleCount; i++) {
        const bubbleGeometry = new THREE.SphereGeometry(0.01, 8, 8);
        const bubbleMaterial = new THREE.MeshBasicMaterial({
            color: "#ffffff",
            transparent: true,
            opacity: 0.3
        });
        
        const bubble = new THREE.Mesh(bubbleGeometry, bubbleMaterial);
        bubble.position.set(
            Math.random() * length - length/2,
            Math.random() * 0.02,
            Math.random() * width - width/2
        );
        
        bubble.userData = {
            // speed: 0.02 + Math.random() * 0.01,
            speed: speed,
            originalY: bubble.position.y
        };
        
        bubbleGroup.add(bubble);
        bubbles.push(bubble);
    }
    
    scene.add(bubbleGroup);
    
    return {
        update: function() {
            const time = performance.now() * 0.001;
            
            bubbles.forEach((bubble, index) => {
                // 气泡向上移动
                bubble.position.y += bubble.userData.speed;
                
                // 轻微的水平摆动
                bubble.position.x += Math.sin(time * 2 + index) * 0.001;
                
                // 如果气泡超出范围，重置位置
                if (bubble.position.y > 0.02) {
                    bubble.position.y = bubble.userData.originalY;
                    bubble.position.x = Math.random() * length - length/2;
                }
                
                // 气泡大小变化
                const scale = 0.8 + Math.sin(time * 3 + index) * 0.2;
                bubble.scale.set(scale, scale, scale);
            });
        },
        bubbleGroup: bubbleGroup
    };
}

export function createWaveWater(options = {}) {
  // 波浪控制参数
  const {
    waveHeight = 0.05,        // 波浪高度
    waveFrequency = 4.0,       // 波浪频率
    waveSpeed = 2.0,           // 波浪速度
    textureSpeed = 0.03,       // 纹理动画速度
    enableLargeWave = false,    // 是否启用大尺度波浪
    largeWaveHeight = 0.08,    // 大尺度波浪高度
    largeWaveFrequency = 1.5,  // 大尺度波浪频率
    enableFlowWave = true,      // 是否启用流动波浪
    flowWaveHeight = 0.04,     // 流动波浪高度
    flowWaveFrequency = 6.0,   // 流动波浪频率
    subdivisions = { width: 128, height: 64 }, // 几何体细分
    pipeLength = 5,
    height = 0.23,
    flowDirection = 1,         // 流动方向：1表示向右，-1表示向左
    waveSourceX = -2.5,        // 波浪源点X坐标（管道左侧）
    wavePropagation = true     // 是否启用波浪传播效果
  } = options;

  // 创建水面几何体 - 根据参数调整细分
  const geometry = new THREE.PlaneGeometry(
    pipeLength,
    height, 
    subdivisions.width, 
    subdivisions.height
  );
  
  // 使用内置材质，但添加纹理流动效果
  const waterMaterial = new THREE.MeshPhongMaterial({
    color: "#4488ff",
    transparent: true,
    opacity: 0.6,
    specular: "#ffffff",
    shininess: 100,
    reflectivity: 0.5
  });

  // 创建水面网格
  const water = new THREE.Mesh(geometry, waterMaterial);
  water.rotation.x = -Math.PI / 2;
  water.position.set(0, 0.01, 0);
  
  // 创建流动纹理 - 向右流动
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');
  
  // 绘制流动纹理 - 向右流动
  function createFlowTexture(time) {
    ctx.fillStyle = '#4488ff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制流动条纹 - 向右流动
    ctx.strokeStyle = '#74ccf4';
    ctx.lineWidth = 5;
    ctx.globalAlpha = 0.8;
    
    for (let i = 0; i < 30; i++) {
      // 向右流动：时间乘以正数，条纹向右移动
      const offset = (time * 80 * textureSpeed * 10 * flowDirection + i * 20) % canvas.width;
      ctx.beginPath();
      ctx.moveTo(offset, 0);
      ctx.bezierCurveTo(
        offset + 80, canvas.height / 3,
        offset - 50, canvas.height * 2 / 3,
        offset, canvas.height
      );
      ctx.stroke();
    }
    
    ctx.globalAlpha = 1.0;
    return new THREE.CanvasTexture(canvas);
  }
  
  // 顶点动画实现流动效果 - 向右流动
  const vertices = geometry.attributes.position.array;
  const originalPositions = new Float32Array(vertices);
  let textureTime = 0;
  let waveStartTime = performance.now() * 0.001;
  
  function updateWaterAnimation() {
    const time = performance.now() * 0.001;
    textureTime += textureSpeed;
    
    // 更新纹理 - 向右流动
    waterMaterial.map = createFlowTexture(textureTime);
    waterMaterial.needsUpdate = true;
    
    // 顶点动画 - 实现波浪传播效果
    for (let i = 0; i < vertices.length; i += 3) {
      const x = vertices[i];
      const z = vertices[i + 2];
      
      let totalWave = 0;
      
      if (wavePropagation) {
        // 波浪传播效果：波浪从源点向远处传播
        const distanceFromSource = Math.abs(x - waveSourceX);
        const waveArrivalTime = distanceFromSource / waveSpeed;
        
        // 主波浪 - 传播效果
        const wave1 = Math.sin(waveFrequency * (x - waveSourceX) - (time - waveArrivalTime) * waveSpeed * flowDirection) * waveHeight;
        
        // 横向波浪 - 传播效果
        const wave2 = Math.cos(waveFrequency * 0.75 * (x - waveSourceX) - (time - waveArrivalTime) * waveSpeed * 0.75 * flowDirection) * (waveHeight * 0.6);
        
        // 斜向波浪 - 传播效果
        const wave3 = Math.sin(waveFrequency * 0.5 * (x - waveSourceX + z) - (time - waveArrivalTime) * waveSpeed * 0.5 * flowDirection) * (waveHeight * 0.4);
        
        totalWave = wave1 + wave2 + wave3;
        
        // 流动效果 - 传播波浪
        if (enableFlowWave) {
          const flowWave = Math.sin(flowWaveFrequency * (x - waveSourceX) - (time - waveArrivalTime) * waveSpeed * 1.5 * flowDirection) * flowWaveHeight;
          totalWave += flowWave;
        }
        
        // 大尺度波浪 - 传播效果
        if (enableLargeWave) {
          const largeWave = Math.sin(largeWaveFrequency * (x - waveSourceX) - (time - waveArrivalTime) * waveSpeed * 0.4 * flowDirection) * largeWaveHeight;
          totalWave += largeWave;
        }
        
        // 波浪衰减：距离源点越远，波浪越小
        const attenuation = Math.max(0, 1 - distanceFromSource / (pipeLength * 0.8));
        totalWave *= attenuation;
      } else {
        // 传统波浪效果（保持向后兼容）
        const wave1 = Math.sin(x * waveFrequency - time * waveSpeed * flowDirection) * waveHeight;
        const wave2 = Math.cos(z * (waveFrequency * 0.75) - time * (waveSpeed * 0.75) * flowDirection) * (waveHeight * 0.6);
        const wave3 = Math.sin((x + z) * (waveFrequency * 0.5) - time * (waveSpeed * 0.5) * flowDirection) * (waveHeight * 0.4);
        
        totalWave = wave1 + wave2 + wave3;
        
        if (enableFlowWave) {
          const flowWave = Math.sin(x * flowWaveFrequency - time * waveSpeed * 1.5 * flowDirection) * flowWaveHeight;
          totalWave += flowWave;
        }
        
        if (enableLargeWave) {
          const largeWave = Math.sin(x * largeWaveFrequency - time * waveSpeed * 0.4 * flowDirection) * largeWaveHeight;
          totalWave += largeWave;
        }
      }
      
      vertices[i + 1] = originalPositions[i + 1] + totalWave;
    }
    
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();
  }

  // 波浪参数控制方法
  function setWaveParams(newParams) {
    Object.keys(newParams).forEach(key => {
      if (options.hasOwnProperty(key)) {
        options[key] = newParams[key];
      }
    });
  }

  // 获取当前波浪参数
  function getWaveParams() {
    return { ...options };
  }

  // 设置流动方向
  function setFlowDirection(direction) {
    options.flowDirection = direction;
  }

  // 设置波浪源点
  function setWaveSource(x) {
    options.waveSourceX = x;
    waveStartTime = performance.now() * 0.001;
  }

  // 启用/禁用波浪传播
  function setWavePropagation(enabled) {
    options.wavePropagation = enabled;
  }

  return { 
    water: water, 
    waterMaterial: waterMaterial,
    updateWaterAnimation: updateWaterAnimation,
    setWaveParams: setWaveParams,
    getWaveParams: getWaveParams,
    setFlowDirection: setFlowDirection,
    setWaveSource: setWaveSource,
    setWavePropagation: setWavePropagation
  };
}
