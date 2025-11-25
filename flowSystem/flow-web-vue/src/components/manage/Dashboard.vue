<template>
  <div class="dashboard-container">
    <!-- 管道监测图区域 -->
    <div class="pipeline-section">
      <div class="section-header">
        <h3>油田管道流型监测系统</h3>
        <div class="controls">
          <el-button 
            type="success" 
            :icon="Refresh" 
            @click="refreshAllPoints"
            :loading="refreshing"
            size="small"
          >
            刷新所有监测点
          </el-button>
        </div>
      </div>

      <div class="pipeline-container">
        <!-- 完善的油田管道示意图 -->
        <div class="pipeline-image" ref="pipelineRef">
          <img src="/img/pipeline.png" alt="油田管道示意图" class="pipeline" />
          
          <!-- 监测点标记 - 精确定位在管道上 -->
          <div 
            v-for="point in monitoringPoints" 
            :key="point.id" 
            class="monitoring-point" 
            :class="{ 
              'active': point.active,
              'normal': point.status === 'normal',
              'warning': point.status === 'warning',
              'error': point.status === 'error'
            }"
            :style="{ 
              left: point.position.x + '%', 
              top: point.position.y + '%' 
            }" 
            @click="handlePointClick(point)" 
          > 
            <div class="point-marker"> 
              <el-icon v-if="point.status === 'normal'"><SuccessFilled /></el-icon> 
              <el-icon v-else-if="point.status === 'warning'"><WarningFilled /></el-icon> 
              <el-icon v-else><CircleCloseFilled /></el-icon> 
            </div> 
            <div class="point-tooltip"> 
              {{ point.name }} 
            </div> 
          </div> 
        </div> 

        <!-- 右侧监测点信息 -->
        <div class="points-sidebar">
          <div class="sidebar-header">
            <h4>监测点列表</h4>
          </div>
          
          <div class="points-list">
            <div 
              v-for="point in monitoringPoints" 
              :key="point.id" 
              class="point-item" 
              :class="{ 
                'active': selectedPoint?.id === point.id, 
                [point.status]: true
              }" 
              @click="selectPoint(point)" 
            > 
              <div class="point-info"> 
                <div class="point-name">{{ point.name }}</div> 
                <div class="point-location">位置: {{ point.location }}</div> 
                <div class="point-device">设备: {{ point.deviceId }}</div> 
              </div> 
              <div class="point-status"> 
                <el-tag 
                  :type="getStatusType(point.status)" 
                  size="small" 
                > 
                  {{ getStatusText(point.status) }} 
                </el-tag> 
              </div> 
            </div> 
          </div> 
        </div> 
      </div>
    </div>

    <!-- 监测结果对话框 -->
    <el-dialog
      v-model="resultDialogVisible"
      :title="`监测点 ${selectedPoint?.name} - 检测结果`"
      width="1600px"
    >
      <div v-loading="detectionLoading" class="detection-result">
        <div v-if="detectionResult" class="result-content">
          <div class="result-header">
            <div class="result-status">
              <el-tag :type="getStatusType(detectionResult.status)" size="large">
                {{ detectionResult.statusText }}
              </el-tag>
              <span class="result-time">检测时间: {{ detectionResult.timestamp }}</span>
            </div>
          </div>
          
          <div class="result-metrics">
            <el-row :gutter="20">
              <el-col :span="8">
                <div class="metric-box">
                  <div class="metric-label">流型类型</div>
                  <div class="metric-value">{{ detectionResult.flowType }}</div>
                </div>
              </el-col>
              <el-col :span="8">
                <div class="metric-box">
                  <div class="metric-label">置信度</div>
                  <div class="metric-value">{{ detectionResult.percentage_confidence }}%</div>
                  <el-progress
                    :percentage="detectionResult.percentage_confidence"
                    :status="detectionResult.percentage_confidence > 80 ? 'success' : 'warning'"
                  />
                </div>
              </el-col>
              <el-col :span="8">
                <div class="metric-box">
                  <div class="metric-label">预测类别</div>
                  <div class="metric-value">{{ detectionResult.preLabel }}</div>
                </div>
              </el-col>
            </el-row>
          </div>

          <el-row :gutter="20">
            <el-col :span="12">
          <div class="result-curve">
            <h4>流型特征曲线</h4>
            <canvas class="flowCurveCanvas" ref="curveCanvasRef"></canvas>
          </div>
            </el-col>
          <el-col :span="12">
          <div class="result-anime">
            <h4>流型特征模拟</h4>
            <canvas class="flowAnimationCanvas" ref="animeCanvasRef"></canvas>
          </div>
          </el-col>
          </el-row>
        </div>
        
        <div v-else-if="!detectionLoading" class="no-result">
          <el-empty description="暂无检测数据" />
        </div>
      </div>
      
      <template #footer>
        <div style="display: flex; justify-content: center; gap: 10px;">
          <el-button @click="resultDialogVisible = false">关闭</el-button>
          <el-button 
            type="primary" 
            :icon="Refresh" 
            @click="refreshDetection"
            :loading="detectionLoading"
          >
            重新检测
          </el-button>
          <el-checkbox @change="handleAutoRefresh" label="自动刷新检测"/>
          <el-button 
            type="primary" 
            :icon="Upload" 
            @click="uploadFlowPattern"
          >
            上传流型数据
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, onUnmounted } from 'vue'
import { 
  Refresh,
  SuccessFilled,
  WarningFilled,
  CircleCloseFilled
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { runFlowIdentification } from '@/api/flow-identification'
import { draw_flow_curve } from '@/assets/js/flow-curve'
import { createFlowPattern } from '@/api/flow-pattern'

// 响应式数据
const viewMode = ref('overview')
const onlineStatus = ref(true)
const activePoints = ref(10)
const totalPoints = ref(12)
const avgResponseTime = ref(156)
const refreshing = ref(false)
const resultDialogVisible = ref(false)
const detectionLoading = ref(false)
const selectedPoint = ref(null)
const detectionResult = ref(null)
const pipelineRef = ref(null)
// 添加 canvas 引用
const curveCanvasRef = ref(null)
const animeCanvasRef = ref(null)

// 监测点数据 - 精确定位在管道上
const monitoringPoints = ref([
  {
    id: 5,
    name: '主管道连接',
    location: '主管道连接处',
    deviceId: 'RK3588-MAIN-004',
    position: { x: 17, y: 50 },
    status: 'normal',
    active: true,
    address: 'http://172.25.157.31:5000'
  },
  {
    id: 6,
    name: '水平主管道',
    location: '水平主管道中段',
    deviceId: 'RK3588-MAIN-002',
    position: { x: 55, y: 75 },
    status: 'normal',
    active: true,
    address: 'http://172.25.157.31:5000'
  },
  {
    id: 7,
    name: '弯道监测点',
    location: '主管道弯道处',
    deviceId: 'RK3588-MAIN-003',
    position: { x: 56, y: 47 },
    status: 'warning',
    active: true,
    address: 'http://172.25.157.31:5000'
  },
  {
    id: 8,
    name: '上倾监测点',
    location: '主管道上倾管处',
    deviceId: 'RK3588-PRES-001',
    position: { x: 67, y: 33 },
    status: 'normal',
    active: true,
    address: 'http://172.25.157.31:5000'
  },
  {
    id: 9,
    name: '上部管道',
    location: '上部管道处',
    deviceId: 'RK3588-SEP-001',
    position: { x: 50, y: 5 },
    status: 'normal',
    active: true,
    address: 'http://172.25.157.31:5000'
  },
  {
    id: 11,
    name: '泵站后监测',
    location: '增压泵站后管道',
    deviceId: 'RK3588-PUMP-002',
    position: { x: 75, y: 39 },
    status: 'warning',
    active: true
  },
  {
    id: 12,
    name: '输送管道',
    location: '输送管道处',
    deviceId: 'RK3588-TANK-001',
    position: { x: 83, y: 25 },
    status: 'error',
    active: false,
    address: 'http://172.25.157.31:5000'
  }
])


// 计算属性
const responseTimeTrend = computed(() => {
  return avgResponseTime.value < 200 ? 'positive' : 'negative'
})

// 方法
const getStatusType = (status) => {
  const map = {
    normal: 'success',
    warning: 'warning',
    error: 'danger'
  }
  return map[status] || 'info'
}

const getStatusText = (status) => {
  const map = {
    normal: '正常',
    warning: '警告',
    error: '异常'
  }
  return map[status] || '未知'
}

const selectPoint = (point) => {
  selectedPoint.value = point
}

// 向RK3588设备发送请求
const sendDetectionRequest = async (point) => {
  const data = { sample_length: 4096, step: 2048 };
  return runFlowIdentification(point.address, data)
}

const handlePointClick = async (point) => {
  selectedPoint.value = point
  resultDialogVisible.value = true
  await refreshDetection()
}

let flowAnimeModule = null;

const refreshDetection = async () => {
  if (!selectedPoint.value) return
  
  detectionLoading.value = true;

  try {
    // 模拟向RK3588设备发送检测请求
    const data = await sendDetectionRequest(selectedPoint.value);
    let result = data.results[0];
    result.percentage_confidence = (result.confidence * 100).toFixed(2);
    result.status = result.confidence > 0.8 ? 'normal' : (result.confidence > 0.5 ? 'warning' : 'error');
    result.statusText = getStatusText(result.status);
    result.timestamp = new Date().toLocaleString();

    detectionResult.value = result;

    // 等待对话框完全渲染后再绘制曲线
    await nextTick();

    if (curveCanvasRef.value) {
      draw_flow_curve(curveCanvasRef.value, result.flowData);
    } else {
      console.warn('curve: Canvas element not found');
    }

    // 停止之前的动画
    if (flowAnimeModule && flowAnimeModule.stopAnimation) {
      flowAnimeModule.stopAnimation();
    }
    if(animeCanvasRef.value){
      import('@/assets/js/flow-anime.js').then(module => {
        flowAnimeModule = module;
        flowAnimeModule.default(result.preLabel); // 传入流型类型
      }).catch(err => {
        console.error('Failed to load flow-anime module:', err);
      });
    } else {
      console.warn('anime: Canvas element not found');
    }

    ElMessage.success(`监测点 ${selectedPoint.value.name} 检测完成`)
  } catch (error) {
    ElMessage.error('检测请求失败')
    console.error('Detection error:', error)
  } finally {
    detectionLoading.value = false
  }
}

let refreshInterval = null;
const handleAutoRefresh = (checked) => {
  if (checked) {
    autoRefreshDetection();
    refreshInterval = setInterval(autoRefreshDetection, 3000); // 每3秒刷新一次
  } else {
    clearInterval(refreshInterval)
  }
}

const autoRefreshDetection = async () => {
  if (!selectedPoint.value) return
  
  try {
    // 模拟向RK3588设备发送检测请求
    const data = await sendDetectionRequest(selectedPoint.value);
    let result = data.results[0];
    result.percentage_confidence = (result.confidence * 100).toFixed(2);
    result.status = result.confidence > 0.8 ? 'normal' : (result.confidence > 0.5 ? 'warning' : 'error');
    result.statusText = getStatusText(result.status);
    result.timestamp = new Date().toLocaleString();

    detectionResult.value = result;

    // 等待对话框完全渲染后再绘制曲线
    await nextTick();

    if (curveCanvasRef.value) {
      draw_flow_curve(curveCanvasRef.value, result.flowData);
    } else {
      console.warn('curve: Canvas element not found');
    }

    // 停止之前的动画
    if (flowAnimeModule && flowAnimeModule.stopAnimation) {
      flowAnimeModule.stopAnimation();
    }
    if(animeCanvasRef.value){
      import('@/assets/js/flow-anime.js').then(module => {
        flowAnimeModule = module;
        flowAnimeModule.default(result.preLabel); // 传入流型类型
      }).catch(err => {
        console.error('Failed to load flow-anime module:', err);
      });
    } else {
      console.warn('anime: Canvas element not found');
    }

    ElMessage.success(`监测点 ${selectedPoint.value.name} 检测完成`)
  } catch (error) {
    ElMessage.error('检测请求失败')
    console.error('Detection error:', error)
  } finally {
  }
}

const refreshAllPoints = async () => {
  refreshing.value = true
  try {
    // 模拟批量刷新所有监测点
    await new Promise(resolve => setTimeout(resolve, 2000))
    ElMessage.success('所有监测点已刷新')
  } catch (error) {
    ElMessage.error('刷新失败')
  } finally {
    refreshing.value = false
  }
}

onMounted(() => {
  // 初始化数据
})

onUnmounted(() => {
  // 组件卸载时停止动画
  if (flowAnimeModule && flowAnimeModule.stopAnimation) {
    flowAnimeModule.stopAnimation();
  }
})

const uploadFlowPattern = async () => {
  const data = {
    patternType: detectionResult.value.flowType,
    predictedLabel: detectionResult.value.preLabel,
    prob: parseFloat(detectionResult.value.confidence),
    data: detectionResult.value.flowData,
    addr: selectedPoint.value.location  // 检测地点
  };
  const res = await createFlowPattern(data);
  console.log('上传流型数据结果:', res);
  if(res.code == 0){
    ElMessage.success('流型数据上传成功');
  } else {
    ElMessage.error('流型数据上传失败');
  }
}
</script>

<style scoped>
.dashboard-container {
  background-color: #f5f7fa;
  margin: 0;
}


/* 管道监测区域 */
.pipeline-section {
  background: white;
  border-radius: 8px;
  padding: 20px;
  padding-bottom: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h3 {
  margin: 0;
  color: #303133;
}

.controls {
  display: flex;
  gap: 12px;
  align-items: center;
}

.pipeline-container {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 15px;  /* 减小间距 */
  height: 666px;
}

.pipeline-image {
  position: relative;
  border: 2px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.pipeline {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  /* border: 1px solid #1261ff; */
}

.points-sidebar {
  border-left: 1px solid #e4e7ed;
  padding-left: 12px;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.sidebar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  padding: 6px 0;
  min-height: 32px;  /* 设置最小高度 */
}

.sidebar-header h4 {
  margin: 0;
  color: #303133;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 120px;  /* 限制标题宽度 */
}

/* 优化监测点列表 */
.points-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  overflow-y: auto;
  max-height: 600px;
  }

.point-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 8px;  /* 进一步减小内边距 */
  border: 1px solid #e4e7ed;
  border-radius: 4px;  /* 减小圆角 */
  cursor: pointer;
  transition: all 0.3s ease;
  min-height: 40px;  /* 设置最小高度 */
}

.point-info {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  margin-right: 6px;  /* 添加右边距 */
}

.point-name {
  font-weight: bold;
  color: #303133;
  margin-bottom: 1px;
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.point-location,
.point-device {
  font-size: 12px;
  color: #909399;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.point-item:hover {
  border-color: #409EFF;
  background: #f0f7ff;
}

.point-item.active {
  border-color: #409EFF;
  background: #ecf5ff;
}

.point-status {
  flex-shrink: 0;
  margin-left: 6px;
}

/* 监测点样式 */
.monitoring-point {
  position: absolute;
  transform: translate(-50%, -50%);
  cursor: pointer;
  transition: all 0.3s ease;
  z-index: 10;
}

.monitoring-point:hover {
  transform: translate(-50%, -50%) scale(1.2);
}

.point-marker {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  border: 2px solid white;
}

.monitoring-point.normal .point-marker {
  background: #67C23A;
}

.monitoring-point.warning .point-marker {
  background: #E6A23C;
}

.monitoring-point.error .point-marker {
  background: #F56C6C;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
}

.monitoring-point.active .point-marker {
  border: 2px solid #409EFF;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.6);
  animation: active-pulse 2s infinite;
}
/* 活跃状态动画 */
@keyframes active-pulse {
  0% { box-shadow: 0 0 8px rgba(64, 158, 255, 0.4); }
  50% { box-shadow: 0 0 15px rgba(64, 158, 255, 0.8); }
  100% { box-shadow: 0 0 8px rgba(64, 158, 255, 0.4); }
}

.point-tooltip {
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
  opacity: 0;
  transition: opacity 0.3s ease;
  pointer-events: none;
  z-index: 20;
}

.monitoring-point:hover .point-tooltip {
  opacity: 1;
}

/* 检测结果样式 */
.detection-result {
  min-height: 300px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid #e4e7ed;
}

.result-time {
  font-size: 12px;
  color: #909399;
}

.result-metrics {
  margin-bottom: 20px;
}

.metric-box {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  text-align: center;
}

.metric-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 8px;
}

.metric-value {
  font-size: 18px;
  font-weight: bold;
  color: #303133;
  margin-bottom: 8px;
}

.result-curve, .result-anime {
  margin-top: 20px;
  width: 100%;
  height: 350px;
}

.result-curve h4, .result-anime h4 {
  margin-bottom: 12px;
  color: #303133;
}
.flowCurveCanvas{
  width:100% !important;  /* 强制宽度为100% */
  height: calc(100% - 50px) !important;
  display: block;  /* 确保canvas是块级元素 */
  margin: 0;
  padding: 0;
  border: 1px solid #04e004;
}
.flowAnimationCanvas {
  width: 100% !important;  /* 强制宽度为100% */
  height: calc(100% - 50px) !important;
  display: block;  /* 确保canvas是块级元素 */
  margin: 0;
  padding: 0;
  background-color: #0b1220;
}

.image-container {
  text-align: center;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 6px;
}

.image-container img {
  max-width: 100%;
  max-height: 200px;
  border-radius: 4px;
  border: 1px solid #e4e7ed;
}

/* 统计区域 */
.stats-section {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.stat-card {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
}

.stat-card h4 {
  margin: 0 0 16px 0;
  color: #303133;
}

.chart-placeholder {
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #e9ecef;
  border-radius: 4px;
  color: #6c757d;
}

.placeholder-text {
  text-align: center;
}
.data-stream {
  height: 200px;
  overflow-y: auto;
}

.stream-item {
  padding: 8px 0;
  border-bottom: 1px solid #e9ecef;
}

.stream-time {
  font-size: 12px;
  color: #6c757d;
  margin-bottom: 4px;
}

.stream-content {
  font-size: 14px;
  color: #495057;
}
</style>