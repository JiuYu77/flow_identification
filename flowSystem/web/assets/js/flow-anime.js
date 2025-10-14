// 专业气液两相流流型动画系统 - 基于实际物理特征设计
// 流型动画系统 - 7种流型可视化

// 段塞流：
// 伪段塞流
// 分层波浪流
// 分层光滑流
// 泡沫段塞流
// 分层泡沫波浪流
// 泡沫环状流

class FlowAnimationSystem {
constructor() {
        this.canvas = null;
        this.ctx = null;
        this.animationId = null;
        this.currentFlowType = '';
        this.particles = [];
        this.time = 0;
        this.flowDirection = 'right';

        // 基于实际物理特征的流型配置
        // 流型配置
        this.flowConfigs = {
            '段塞流': {
                color: '#1E90FF', // 液体蓝色
                density: 45,
                speed: 0.6,
                pattern: 'slug',
                description: '堵塞管道，液塞后有推着液塞前进的气弹'
            },
            '伪段塞流': {
                color: '#1E90FF', // 液体蓝色
                density: 40,
                speed: 0.6,
                pattern: 'pseudoSlug',
                description: '不稳定液桥，介于分层和段塞之间'
            },
            '分层波浪流': {
                color: '#1E90FF', // 液体蓝色
                density: 35,
                speed: 0.6,
                pattern: 'wavy',
                description: '气液分层，界面有连续运动波浪',
                features: { // 添加缺失的特征属性
                    waveAmplitude: 5
                }
            },
            '分层光滑流': {
                color: '#1E90FF', // 液体蓝色
                density: 30,
                speed: 0.6,
                pattern: 'smooth',
                description: '气液完全分离，界面平整光滑'
            },
            '泡沫段塞流': {
                liquidColor: '#1E90FF', // 液体部分蓝色
                foamColor: '#FFD700',   // 泡沫部分金色
                density: 50,
                speed: 0.6,
                pattern: 'foamSlug',
                description: '段塞流形态，液塞中存在泡沫'
            },
            '分层泡沫波浪流': {
                liquidColor: '#1E90FF', // 液体部分蓝色
                foamColor: '#FFD700',   // 泡沫部分金色
                density: 42,
                speed: 0.6,
                pattern: 'foamWavy',
                description: '分层流，液相有泡沫，界面有波浪'
            },
            '泡沫环状流': {
                liquidColor: '#1E90FF', // 液体部分蓝色
                foamColor: '#FFD700',   // 泡沫部分金色
                density: 55,
                speed: 0.6,
                pattern: 'foamAnnular',
                description: '管壁液膜充满泡沫，中心气体流动'
            }
        };
    }

    // 初始化动画系统
    init(canvasElement, direction = 'right') {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.flowDirection = direction;
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    // 调整画布大小
    resizeCanvas() {
        if (!this.canvas) return;

        const container = this.canvas.parentElement;
        if (container) {
            this.canvas.width = container.clientWidth;
            this.canvas.height = container.clientHeight;
        }
    }

    // 开始特定流型的动画
    startFlowAnimation(flowType, direction = 'right') {
        this.stopAnimation();
        this.currentFlowType = flowType;
        this.flowDirection = direction;
        this.particles = [];
        this.time = 0;
        
        const config = this.flowConfigs[flowType];
        if (!config) {
            console.error('未知的流型:', flowType);
            return;
        }

        // 创建基于物理特征的粒子系统
        this.createPhysicalParticles(config);
        
        // 开始动画循环
        this.animate();
    }

    // 基于物理特征创建粒子
    createPhysicalParticles(config) {
        const directionMultiplier = this.flowDirection === 'right' ? 1 : -1;
        const particleCount = Math.floor((this.canvas.width * this.canvas.height) / 400 * config.density / 50);
        
        // 根据流型特征创建不同的粒子分布
        switch(config.pattern) {
            case 'slug': // 段塞流：液塞和气弹交替
                this.createSlugFlowParticles(config, particleCount, directionMultiplier);
                break;
            case 'pseudoSlug': // 伪段塞流：不稳定波浪
                this.createPseudoSlugParticles(config, particleCount, directionMultiplier);
                break;
            case 'wavy': // 分层波浪流：分层波浪界面
                this.createWavyStratifiedParticles(config, particleCount, directionMultiplier);
                break;
            case 'smooth': // 分层光滑流：平稳分层
                this.createSmoothStratifiedParticles(config, particleCount, directionMultiplier);
                break;
            case 'foamSlug': // 泡沫段塞流：泡沫液塞
                this.createFoamSlugParticles(config, particleCount, directionMultiplier);
                break;
            case 'foamWavy': // 分层泡沫波浪流：泡沫波浪
                this.createFoamWavyParticles(config, particleCount, directionMultiplier);
                break;
            case 'foamAnnular': // 泡沫环状流：环状泡沫
                this.createFoamAnnularParticles(config, particleCount, directionMultiplier);
                break;
        }
    }

    // 段塞流粒子创建：液塞和气弹交替
    // 段塞流粒子创建：堵塞管道，液塞后有举着液塞前进的气弹
    createSlugFlowParticles(config, count, directionMultiplier) {
        this.particles = [];
        
        // 创建交替的液塞和气弹
        const segments = [];
        let currentPosition = this.flowDirection === 'right' ? -this.canvas.width * 0.5 : this.canvas.width * 1.5;
        const totalSegments = 4;
        
        for (let i = 0; i < totalSegments; i++) {
            const isLiquidSegment = i % 2 === 0;
            const segmentLength = isLiquidSegment ? this.canvas.width * 0.5 : this.canvas.width * 0.3;
            
            segments.push({
                startX: currentPosition,
                endX: currentPosition + segmentLength * directionMultiplier,
                isLiquid: isLiquidSegment,
                segmentIndex: i
            });
            
            currentPosition += segmentLength * directionMultiplier;
        }
        
segments.forEach(segment => {
            const particlesPerSegment = Math.floor(count / segments.length * 1.2);
            
for (let i = 0; i < particlesPerSegment; i++) {
                const segmentProgress = Math.random();
                const initialX = segment.startX + segmentProgress * (segment.endX - segment.startX);
                
                if (segment.isLiquid) {
                    // 液塞：几乎充满管道下部（70-95%高度）
                    const initialY = this.canvas.height * 0.7 + Math.random() * this.canvas.height * 0.25;
                    
                    this.particles.push({
                        x: initialX,
                        y: initialY,
                        size: 2.0 + Math.random() * 1.0,
                        speed: config.speed,
                        direction: directionMultiplier,
                        type: config.pattern,
                        isLiquidSegment: true,
                        segmentIndex: segment.segmentIndex,
                        segmentProgress: segmentProgress,
                        slugPhase: Math.random() * Math.PI * 2, // 添加缺失的属性
                        turbulence: 0.5 + Math.random() * 0.5   // 添加缺失的属性
                    });
                } else {
                    // 气弹：少量液体被气体携带（30-50%高度）
                    const initialY = this.canvas.height * 0.3 + Math.random() * this.canvas.height * 0.2;
                    
                    this.particles.push({
                        x: initialX,
                        y: initialY,
                        size: 1.0 + Math.random() * 0.5,
                        speed: config.speed,
                        direction: directionMultiplier,
                        type: config.pattern,
                        isLiquidSegment: false,
                        segmentIndex: segment.segmentIndex,
                        segmentProgress: segmentProgress
                    });
                }
            }
        });
    }

    // 段塞流粒子创建：堵塞管道，液塞后有推着液塞前进的气弹
    createSlugFlowParticles(config, count, directionMultiplier) {
        this.particles = [];
        
        // 创建更真实的段塞流：液塞几乎充满管道，气弹推动前进
        const segments = [];
        let currentPosition = this.flowDirection === 'right' ? -this.canvas.width * 0.5 : this.canvas.width * 1.5;
        const totalSegments = 3; // 减少段数，增加每个液塞的长度

        for (let i = 0; i < totalSegments; i++) {
            const isLiquidSegment = i % 2 === 0;
            // 液塞更长，几乎充满管道；气弹较短
            const segmentLength = isLiquidSegment ? this.canvas.width * 0.7 : this.canvas.width * 0.2;
            
            segments.push({
                startX: currentPosition,
                endX: currentPosition + segmentLength * directionMultiplier,
                isLiquid: isLiquidSegment,
                segmentIndex: i
            });
            
            currentPosition += segmentLength * directionMultiplier;
        }
        
        segments.forEach(segment => {
            const particlesPerSegment = Math.floor(count / segments.length * 1.5);
            
            for (let i = 0; i < particlesPerSegment; i++) {
                const segmentProgress = Math.random();
                const initialX = segment.startX + segmentProgress * (segment.endX - segment.startX);
                
                if (segment.isLiquid) {
                    // 液塞：几乎充满管道（80-95%高度），体现堵塞特征
                    const initialY = this.canvas.height * 0.8 + Math.random() * this.canvas.height * 0.15;
                    
                    this.particles.push({
                        x: initialX,
                        y: initialY,
                        size: 2.5 + Math.random() * 1.0, // 更大的粒子体现密集
                        speed: config.speed,
                        direction: directionMultiplier,
                        type: config.pattern,
                        isLiquidSegment: true,
                        segmentIndex: segment.segmentIndex,
                        segmentProgress: segmentProgress,
                        slugPhase: Math.random() * Math.PI * 2,
                        turbulence: 0.8 + Math.random() * 0.4 // 更强的湍流
                    });
                } else {
                    // 气弹：少量液体被气体携带（20-40%高度）
                    const initialY = this.canvas.height * 0.2 + Math.random() * this.canvas.height * 0.2;
                    
                    this.particles.push({
                        x: initialX,
                        y: initialY,
                        size: 1.0 + Math.random() * 0.3, // 更小的粒子
                        speed: config.speed,
                        direction: directionMultiplier,
                        type: config.pattern,
                        isLiquidSegment: false,
                        segmentIndex: segment.segmentIndex,
                        segmentProgress: segmentProgress
                    });
                }
            }
        });
    }

    // 伪段塞流粒子创建：连续液体流动，低液面与几乎堵塞管道的高液面相间出现
    createPseudoSlugParticles(config, count, directionMultiplier) {
        this.particles = [];
        
        // 伪段塞流特征：连续的液体流动，高低液面相间
        // 预先生成连续的液体分布，然后整体流动
        const highLevelHeight = this.canvas.height * 0.8; // 高液面高度（几乎堵塞）
        const lowLevelHeight = this.canvas.height * 0.3;  // 低液面高度
        const segmentLength = this.canvas.width * 0.6;   // 每个液段的长度
        
        // 创建连续的液体分布
        const totalSegments = 3; // 总共3个液段（高低交替）
        
        for (let seg = 0; seg < totalSegments; seg++) {
            const isHighLevel = seg % 2 === 0;
            const liquidHeight = isHighLevel ? highLevelHeight : lowLevelHeight;
            const segmentStartX = seg * segmentLength;
            const segmentEndX = (seg + 1) * segmentLength;
            
            // 每个液段的粒子数量
const particlesPerSegment = Math.floor(count / totalSegments);
            
for (let i = 0; i < particlesPerSegment; i++) {
                // 在液段内均匀分布粒子
                const segmentProgress = i / particlesPerSegment;
                const initialX = segmentStartX + segmentProgress * segmentLength;
                
                // 粒子在液体中的垂直分布：从底部向上
                const initialY = this.canvas.height - (liquidHeight * Math.random());
                
                // 粒子大小相对均匀
                const particleSize = 1.5 + Math.random() * 0.5;
                
                this.particles.push({
                    x: initialX,
                    y: initialY,
                    size: particleSize,
                    speed: config.speed,
                    direction: directionMultiplier,
                    type: config.pattern,
                    isHighLevel: isHighLevel,
                    liquidHeight: liquidHeight,
                    segmentIndex: seg,
                    baseX: initialX // 记录基础位置用于循环
                });
            }
        }
        
        // 添加一些额外的粒子来增强连续性
        const extraParticles = count - this.particles.length;
        for (let i = 0; i < extraParticles; i++) {
            const initialX = Math.random() * totalSegments * segmentLength;
            const seg = Math.floor(initialX / segmentLength) % totalSegments;
            const isHighLevel = seg % 2 === 0;
            const liquidHeight = isHighLevel ? highLevelHeight : lowLevelHeight;
            const initialY = this.canvas.height - (liquidHeight * Math.random());
            
            this.particles.push({
                x: initialX,
                y: initialY,
                size: 1.5 + Math.random() * 0.5,
                speed: config.speed,
                direction: directionMultiplier,
                type: config.pattern,
                isHighLevel: isHighLevel,
                liquidHeight: liquidHeight,
                segmentIndex: seg,
                baseX: initialX
            });
        }
    }

    // 分层波浪流粒子创建：类似起波浪的小河
    createWavyStratifiedParticles(config, count, directionMultiplier) {
        this.particles = [];
        
        // 波浪参数
        const waveAmplitude = this.canvas.height * 0.4; // 大幅增加波浪幅度到40%
        const waveFrequency = 0.002; // 降低频率，使波浪更平滑
        const baseHeight = this.canvas.height * 0.6; // 基线高度60%
        
        for (let i = 0; i < count; i++) {
            let initialX, initialY;
            
            // 初始X位置分布
            if (this.flowDirection === 'right') {
                initialX = -Math.random() * this.canvas.width * 0.8;
            } else {
                initialX = this.canvas.width + Math.random() * this.canvas.width * 0.8;
            }
            
            // 基于X位置的波浪相位，确保波浪连续性
            const wavePhase = initialX * waveFrequency;
            const wave = Math.sin(wavePhase) * waveAmplitude;
            
            // 创建波浪效果
            initialY = baseHeight + wave;
            
            // 添加随机高度变化，使波浪更自然
initialY += Math.random() * this.canvas.height * 0.15;
            
            this.particles.push({
                x: initialX,
                y: initialY,
                size: 4.0 + Math.random() * 2.0, // 大幅增大粒子尺寸
                speed: config.speed,
                direction: directionMultiplier,
                type: config.pattern,
                wavePhase: wavePhase,
                baseY: baseHeight,
                waveAmplitude: waveAmplitude,
                waveFrequency: waveFrequency,
                waveSpeed: 0.008 // 降低波浪移动速度
            });
        }
    }

    // 基于物理特征的粒子更新
    updateParticleByPhysicalType(particle, config) {
        const baseSpeed = particle.speed * particle.direction;
        
        switch (particle.type) {
            case 'slug': // 段塞流：强烈的间歇性高速流动
                // 液塞高速前进，体现堵塞管道和向前推进特征
                particle.x += baseSpeed * 4.0; // 更快的速度
            
                // 液塞内部的强烈湍流混合
                particle.y += Math.sin(this.time * 0.3 + particle.x * 0.06 + particle.slugPhase) * particle.turbulence * 3.0;
            
                // 液塞前缘的强烈冲击效应
                if (particle.segmentProgress < 0.15) {
                    particle.x += baseSpeed * 0.5;
                    particle.size = 2.8; // 前缘粒子更大
                }
                
                // 液塞尾部的液体剥离
                if (particle.segmentProgress > 0.85) {
                    particle.y += Math.sin(this.time * 0.4) * 1.0;
                    particle.size = 2.0; // 尾部粒子稍小
                }
                break;

            case 'pseudoSlug': // 伪段塞流：连续液体流动，高低液面相间
                // 基本流动速度
                particle.x += baseSpeed * 1.2;
                
                // 当粒子流出画布时，重新放回起始位置，保持连续流动
                const totalSegments = 3;
                const segmentLength = this.canvas.width * 0.6;
                const totalLength = totalSegments * segmentLength;
                
                if (this.flowDirection === 'right') {
                    if (particle.x > this.canvas.width + segmentLength) {
                        // 循环到起始位置
                        particle.x = -segmentLength + (particle.x - (this.canvas.width + segmentLength));
                    }
                } else {
                    if (particle.x < -segmentLength) {
                        // 循环到结束位置
                        particle.x = this.canvas.width + segmentLength + (particle.x + segmentLength);
                    }
                }
                
                // 根据当前位置更新液段信息
                const currentSegment = Math.floor((particle.x + totalLength) % totalLength / segmentLength);
                const isHighLevel = currentSegment % 2 === 0;
                const liquidHeight = isHighLevel ? (this.canvas.height * 0.8) : (this.canvas.height * 0.3);
                
                // 平滑调整到目标高度
                const targetY = this.canvas.height - (liquidHeight * Math.random());
                const heightDifference = targetY - particle.y;
                particle.y += heightDifference * 0.05; // 缓慢调整高度
                
                // 保持粒子大小相对稳定
                particle.size = 1.5 + Math.random() * 0.3;
                break;

            case 'wavy': // 分层波浪流：类似起波浪的小河
                particle.x += baseSpeed;
                
                // 液体粒子的波浪运动
                const waveAmplitude = particle.waveAmplitude || 20;
                const waveFrequency = particle.waveFrequency || 0.005;
                
                // 使用基于时间和位置的连续波浪运动
                const wave = Math.sin(this.time * 0.02 + particle.wavePhase) * waveAmplitude;
                
                // 围绕基础位置进行波浪运动
                particle.y = particle.baseY + wave;
                
                // 增强波浪湍流效果，使波浪更自然
                const turbulence = Math.sin(this.time * 0.08 + particle.x * 0.02) * 3;
                particle.y += turbulence;
                
                // 确保粒子在画布内
                particle.y = Math.max(0, Math.min(this.canvas.height - particle.size, particle.y));
                break;

            case 'smooth': // 分层光滑流：非常平稳
                particle.x += baseSpeed * particle.smoothness;
                
                // 极小的垂直波动
                if (!particle.isGas) {
                    particle.y += Math.sin(this.time * 0.01) * 0.05;
                }
                break;

            case 'foamSlug': // 泡沫段塞流：高粘度间歇流动
                const foamCycle = (this.time * 0.07 + particle.slugPhase) % (Math.PI * 2);
                if (foamCycle < Math.PI * 1.5) {
                    // 泡沫液塞段
                    particle.x += baseSpeed * 1.8;
                    particle.size = 1.2 + Math.sin(foamCycle) * particle.foamDensity;
                } else {
                    // 气弹段
                    particle.x += baseSpeed * 0.6;
                    particle.size = 1.0;
                }
                
                // 粘度效应：减缓垂直运动
                if (particle.inFoam) {
                    particle.y += Math.sin(this.time * 0.05) * 0.1 * (1 - particle.viscosityEffect);
                }
                break;

            case 'foamWavy': // 分层泡沫波浪流：泡沫波浪界面
                particle.x += baseSpeed;
                
                if (!particle.isGas) {
                    // 泡沫液体的波浪运动
                    const foamWave = Math.sin(this.time * 0.04 + particle.wavePhase) * particle.waveAmplitude;
                    particle.y += foamWave * 0.4;
                    particle.size = 1.0 + Math.sin(this.time * 0.06) * 0.3;
                }
                break;

            case 'foamAnnular': // 泡沫环状流：环状运动
                particle.x += baseSpeed;
                
                if (particle.inLiquidFilm) {
                    // 壁面液膜粒子的环状运动
                    particle.annularAngle += 0.02;
                    const targetY = this.canvas.height / 2 + Math.sin(particle.annularAngle) * 
                                  (this.canvas.height * 0.4 * (0.8 + particle.foamExpansion));
                    particle.y += (targetY - particle.y) * 0.1;
                    particle.size = 1.2 + Math.sin(this.time * 0.05) * 0.4;
                } else {
                    // 中心气核粒子
                    particle.y += Math.sin(this.time * 0.03) * 0.2;
                }
                break;
        }
    }

    // 处理粒子边界 - 保持波浪连续性
    handleParticleBoundary(particle) {
        if (this.flowDirection === 'right') {
            if (particle.x > this.canvas.width + particle.size) {
                particle.x = -particle.size;
                // 对于波浪流，重新计算波浪相位
                if (particle.type === 'wavy') {
                    particle.wavePhase = particle.x * particle.waveFrequency;
                    particle.y = particle.baseY + Math.sin(particle.wavePhase) * particle.waveAmplitude;
                }
            }
        } else {
            if (particle.x < -particle.size) {
                particle.x = this.canvas.width + particle.size;
                // 对于波浪流，重新计算波浪相位
                if (particle.type === 'wavy') {
                    particle.wavePhase = particle.x * particle.waveFrequency;
                    particle.y = particle.baseY + Math.sin(particle.wavePhase) * particle.waveAmplitude;
                }
            }
        }
    }

    // 泡沫环状流粒子创建：管壁液膜充满泡沫，中心气体流动
    createFoamAnnularParticles(config, count, directionMultiplier) {
        this.particles = [];
        
        for (let i = 0; i < count; i++) {
            let initialX, initialY;
            let isFoamBubble = false; // 修复变量作用域问题
            
            if (this.flowDirection === 'right') {
                initialX = -Math.random() * this.canvas.width * 0.15;
            } else {
                initialX = this.canvas.width + Math.random() * this.canvas.width * 0.15;
            }
            
            // 环状流：中心气核，壁面泡沫液膜
            const inLiquidFilm = Math.random() < 0.7;
            const angle = Math.random() * Math.PI * 2;
            const pipeRadius = this.canvas.height * 0.4;
            
            if (inLiquidFilm) {
                // 壁面液膜：充满泡沫气泡
                const filmThickness = 0.15 + Math.random() * 0.1;
                const distanceFromCenter = pipeRadius * (0.85 + filmThickness);
                initialY = this.canvas.height / 2 + Math.sin(angle) * distanceFromCenter;
                
                isFoamBubble = Math.random() < 0.8; // 液膜中80%为泡沫
            } else {
                // 中心气核
                const gasRadius = pipeRadius * 0.7;
                initialY = this.canvas.height / 2 + Math.sin(angle) * gasRadius * Math.random();
            }
            
            this.particles.push({
                x: initialX,
                y: initialY,
                size: inLiquidFilm ? (isFoamBubble ? 1.0 : 1.5) : 1.0,
                speed: config.speed,
                direction: directionMultiplier,
                type: config.pattern,
                inLiquidFilm: inLiquidFilm,
                isFoam: inLiquidFilm && isFoamBubble,
                annularAngle: angle,
                foamExpansion: 0.1 + Math.random() * 0.2 // 添加缺失的属性
            });
        }
    }

    // 更新粒子状态
    updateParticles() {
        const config = this.flowConfigs[this.currentFlowType];
        
        this.particles.forEach(particle => {
            // 根据流型特征更新运动
            this.updateParticleByPhysicalType(particle, config);
            
            // 边界处理
            this.handleParticleBoundary(particle);
        });
    }
    // 基于物理特征的粒子更新
    updateParticleByPhysicalType(particle, config) {
        const baseSpeed = particle.speed * particle.direction;
        
        switch (particle.type) {
            case 'slug': // 段塞流：强烈的间歇性高速流动
                // 液塞高速前进，体现堵塞管道和向前推进特征
                particle.x += baseSpeed * 4.0; // 更快的速度
            
                // 液塞内部的强烈湍流混合
                particle.y += Math.sin(this.time * 0.3 + particle.x * 0.06 + particle.slugPhase) * particle.turbulence * 3.0;
            
                // 液塞前缘的强烈冲击效应
                if (particle.segmentProgress < 0.15) {
                    particle.x += baseSpeed * 0.5;
                    particle.size = 2.8; // 前缘粒子更大
                }
                
                // 液塞尾部的液体剥离
                if (particle.segmentProgress > 0.85) {
                    particle.y += Math.sin(this.time * 0.4) * 1.0;
                    particle.size = 2.0; // 尾部粒子稍小
                }
                break;
            case 'pseudoSlug': // 伪段塞流：连续液体流动，高低液面相间
                // 基本流动速度
                particle.x += baseSpeed * 1.2;
                
                // 当粒子流出画布时，重新放回起始位置，保持连续流动
                const totalSegments = 3;
                const segmentLength = this.canvas.width * 0.6;
                const totalLength = totalSegments * segmentLength;
                
                if (this.flowDirection === 'right') {
                    if (particle.x > this.canvas.width + segmentLength) {
                        // 循环到起始位置
                        particle.x = -segmentLength + (particle.x - (this.canvas.width + segmentLength));
                    }
                } else {
                    if (particle.x < -segmentLength) {
                        // 循环到结束位置
                        particle.x = this.canvas.width + segmentLength + (particle.x + segmentLength);
                    }
                }
                
                // 根据当前位置更新液段信息
                const currentSegment = Math.floor((particle.x + totalLength) % totalLength / segmentLength);
                const isHighLevel = currentSegment % 2 === 0;
                const liquidHeight = isHighLevel ? (this.canvas.height * 0.8) : (this.canvas.height * 0.3);
                
                // 平滑调整到目标高度
                const targetY = this.canvas.height - (liquidHeight * Math.random());
                const heightDifference = targetY - particle.y;
                particle.y += heightDifference * 0.05; // 缓慢调整高度
                
                // 保持粒子大小相对稳定
                particle.size = 1.5 + Math.random() * 0.3;
                break;

            case 'wavy': // 分层波浪流：连续波浪效果
                particle.x += baseSpeed;
                
                // 波浪运动计算 - 使用更简单的波浪公式
                const wave = Math.sin(this.time * particle.waveSpeed + particle.wavePhase) * particle.waveAmplitude;
                
                // 直接设置Y位置，而不是累加
                particle.y = particle.baseY + wave;
                
                // 确保粒子在画布内
                particle.y = Math.max(0, Math.min(this.canvas.height - particle.size, particle.y));
                break;

            case 'smooth': // 分层光滑流：非常平稳
                particle.x += baseSpeed * particle.smoothness;
                
                // 极小的垂直波动
                if (!particle.isGas) {
                    particle.y += Math.sin(this.time * 0.01) * 0.05;
                }
                break;

            case 'foamSlug': // 泡沫段塞流：高粘度间歇流动
                const foamCycle = (this.time * 0.07 + particle.slugPhase) % (Math.PI * 2);
                if (foamCycle < Math.PI * 1.5) {
                    // 泡沫液塞段
                    particle.x += baseSpeed * 1.8;
                    particle.size = 1.2 + Math.sin(foamCycle) * particle.foamDensity;
                } else {
                    // 气弹段
                    particle.x += baseSpeed * 0.6;
                    particle.size = 1.0;
                }
                
                // 粘度效应：减缓垂直运动
                if (particle.inFoam) {
                    particle.y += Math.sin(this.time * 0.05) * 0.1 * (1 - particle.viscosityEffect);
                }
                break;

            case 'foamWavy': // 分层泡沫波浪流：泡沫波浪界面
                particle.x += baseSpeed;
                
                if (!particle.isGas) {
                    // 泡沫液体的波浪运动
                    const foamWave = Math.sin(this.time * 0.04 + particle.wavePhase) * particle.waveAmplitude;
                    particle.y += foamWave * 0.4;
                    particle.size = 1.0 + Math.sin(this.time * 0.06) * 0.3;
                }
                break;

            case 'foamAnnular': // 泡沫环状流：环状运动
                particle.x += baseSpeed;
                
                if (particle.inLiquidFilm) {
                    // 壁面液膜粒子的环状运动
                    particle.annularAngle += 0.02;
                    const targetY = this.canvas.height / 2 + Math.sin(particle.annularAngle) * 
                                  (this.canvas.height * 0.4 * (0.8 + particle.foamExpansion));
                    particle.y += (targetY - particle.y) * 0.1;
                    particle.size = 1.2 + Math.sin(this.time * 0.05) * 0.4;
                } else {
                    // 中心气核粒子
                    particle.y += Math.sin(this.time * 0.03) * 0.2;
                }
                break;
        }
    }

    // 处理粒子边界
    handleParticleBoundary(particle) {
        if (this.flowDirection === 'right') {
            if (particle.x > this.canvas.width + particle.size) {
                particle.x = -particle.size;
                particle.y = Math.random() * this.canvas.height;
            }
        } else {
            if (particle.x < -particle.size) {
                particle.x = this.canvas.width + particle.size;
                particle.y = Math.random() * this.canvas.height;
            }
        }
    }

    // 绘制粒子 - 基于物理特征
    drawParticles() {
        const config = this.flowConfigs[this.currentFlowType];
        
        this.particles.forEach(particle => {
            this.drawPhysicalParticle(particle, config);
        });
    }

    // 绘制基于物理特征的粒子
    drawPhysicalParticle(particle, config) {
        let alpha = 1.0;
        const edgeBuffer = 50;
        
        // 边缘淡入淡出效果
        if (this.flowDirection === 'right') {
            if (particle.x < edgeBuffer) {
                alpha = particle.x / edgeBuffer;
            } else if (particle.x > this.canvas.width - edgeBuffer) {
                alpha = (this.canvas.width - particle.x) / edgeBuffer;
            }
        } else {
            if (particle.x > this.canvas.width - edgeBuffer) {
                alpha = (this.canvas.width - particle.x) / edgeBuffer;
            } else if (particle.x < edgeBuffer) {
                alpha = particle.x / edgeBuffer;
            }
        }
        
        this.ctx.save();
        this.ctx.globalAlpha = alpha;
        
        // 根据流型特征设置颜色
        if (config.liquidColor && config.foamColor) {
            // 泡沫流型：根据粒子类型设置颜色
            if (particle.type === 'foamSlug') {
                this.ctx.fillStyle = particle.inFoam ? config.foamColor : config.liquidColor;
            } else if (particle.type === 'foamWavy') {
                this.ctx.fillStyle = particle.isGas ? config.liquidColor : config.foamColor;
            } else if (particle.type === 'foamAnnular') {
                this.ctx.fillStyle = particle.inLiquidFilm ? config.foamColor : config.liquidColor;
            }
        } else {
            // 普通流型：使用单一颜色
            this.ctx.fillStyle = config.color;
        }
        
        // 根据流型特征绘制不同形状
        switch(particle.type) {
            case 'slug':
            case 'pseudoSlug':
                this.drawSlugParticle(particle);
                break;
            case 'wavy':
            case 'smooth':
                this.drawStratifiedParticle(particle);
                break;
            case 'foamSlug':
            case 'foamWavy':
            case 'foamAnnular':
                this.drawFoamParticle(particle);
                break;
        }
        
        this.ctx.restore();
    }

    // 绘制段塞流粒子
    drawSlugParticle(particle) {
        if (particle.type === 'slug' && particle.isLiquid) {
            // 液塞粒子：较大的矩形
            this.ctx.save();
            this.ctx.translate(particle.x, particle.y);
            this.ctx.rotate(Math.PI * 0.1);
            this.ctx.fillRect(-particle.size, -particle.size/2, particle.size*2, particle.size);
            this.ctx.restore();
        } else {
            // 气弹或其他粒子：圆形
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    // 绘制分层流粒子
    drawStratifiedParticle(particle) {
        if (particle.isGas) {
            // 气体粒子：较小的圆形
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
        } else {
            // 液体粒子：椭圆形，代表界面
            this.ctx.save();
            this.ctx.translate(particle.x, particle.y);
            this.ctx.rotate(Math.PI * 0.05);
            this.ctx.fillRect(-particle.size, -particle.size/3, particle.size*2, particle.size*0.7);
            this.ctx.restore();
        }
    }
    // 绘制泡沫流粒子
    drawFoamParticle(particle) {
        // 泡沫粒子：圆形带光晕效果
        this.ctx.beginPath();
        this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        this.ctx.fill();
        
        // 泡沫光晕效果
        this.ctx.globalAlpha *= 0.3;
        this.ctx.beginPath();
        this.ctx.arc(particle.x, particle.y, particle.size * 1.5, 0, Math.PI * 2);
        this.ctx.fill();
    }

    // 动画循环
    animate() {
        if (!this.ctx || !this.canvas) return;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.updateParticles();
        this.drawParticles();
        this.time++;
        this.animationId = requestAnimationFrame(() => this.animate());
    }

    // 停止动画
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    // 获取流型描述信息
    getFlowTypeDescription(flowType) {
        const config = this.flowConfigs[flowType];
        return config ? config.description : '未知流型';
    }

    // 获取所有支持的流型列表
    getSupportedFlowTypes() {
        return Object.keys(this.flowConfigs);
    }

    // 设置流动方向
    setFlowDirection(direction) {
        if (direction === 'left' || direction === 'right') {
            this.flowDirection = direction;
            if (this.currentFlowType) {
                this.startFlowAnimation(this.currentFlowType, direction);
            }
        }
    }
}

// 全局动画系统实例和辅助函数保持不变
const flowAnimationSystem = new FlowAnimationSystem();

document.addEventListener('DOMContentLoaded', function() {
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1 && node.classList && node.classList.contains('card')) {
                    setupFlowAnimation(node);
                }
            });
        });
    });

    observer.observe(document.getElementById('result'), {
        childList: true,
        subtree: true
    });
});

function setupFlowAnimation(flowType, canvasClassName = '.flow-animation-canvas') {
    const canvas = document.querySelector(canvasClassName);

    flowAnimationSystem.init(canvas, 'right');
    flowAnimationSystem.startFlowAnimation(flowType, 'right');
}

function createFlowAnimation(containerId, flowType, canvasClassName = '.flow-animation-canvas',direction = 'right') {
    const canvas = document.querySelector(canvasClassName);

    const animationSystem = new FlowAnimationSystem();
    animationSystem.init(canvas, direction);
    animationSystem.startFlowAnimation(flowType, direction);
    
    return animationSystem;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = FlowAnimationSystem;
}