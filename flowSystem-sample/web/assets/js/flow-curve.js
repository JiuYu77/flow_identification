/**
 * 在 Canvas 上绘制流动曲线
 * @param {HTMLCanvasElement} canvas - Canvas 元素
 * @param {number[]} data - 流型数据点数组
 * @param {Object} options - 配置选项（可选）
 */
function draw_flow_curve(canvas, data, options = {}) {
  // 默认配置
  const config = {
    lineColor: "#3498db", // 线条颜色
    lineWidth: 2, // 线条宽度
    backgroundColor: "#f8f9fa", // 背景颜色
    gridColor: "#e0e0e0", // 网格颜色
    gridStep: 50, // 网格步长
    padding: {
      // 内边距（分别设置）
      top: 20,
      right: 20, // 右侧增加内边距给Y轴标签留空间
      bottom: 20, // 底部增加内边距给X轴标签留空间
      left: 50, // 左侧增加内边距给Y轴标签留空间
    },
    showGrid: true, // 是否显示网格
    showPoints: false, // 是否显示数据点
    pointRadius: 3, // 数据点半径
    pointColor: "#e74c3c", // 数据点颜色
    smooth: true, // 是否平滑曲线
    ...options, // 用户自定义配置
  };

  // 首先设置画布高分辨率
  const { ctx, logicalWidth, logicalHeight } = setupHighDPICanvas(canvas);
  // 获取画布的逻辑尺寸（CSS像素）
  // logicalWidth
  // logicalHeight

  // 计算实际绘图区域（使用逻辑像素）
  const chartWidth = logicalWidth - config.padding.left - config.padding.right;
  const chartHeight =
    logicalHeight - config.padding.top - config.padding.bottom;

  // 清除画布
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 绘制背景
  ctx.fillStyle = config.backgroundColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 绘制网格
  if (config.showGrid) {
    drawGrid(ctx, logicalWidth, logicalHeight, config.padding, config);
  }

  // 如果没有数据，直接返回
  if (!data || data.length === 0) {
    return;
  }

  // 计算数据范围
  const minValue = Math.min(...data);
  const maxValue = Math.max(...data);
  const valueRange = maxValue - minValue || 1;
  const valueMargin = valueRange * 0.05;

  // 绘制曲线
  drawCurve(
    ctx,
    data,
    minValue - valueMargin,
    valueRange + 2 * valueMargin,
    chartWidth,
    chartHeight,
    config.padding,
    config,
  );

  // 绘制坐标轴标签
  drawAxisLabels(
    ctx,
    logicalWidth,
    logicalHeight,
    config.padding,
    minValue - valueMargin,
    maxValue + valueMargin,
    data.length,
  );
}

/**
 * 设置高DPI Canvas 并返回缩放后的上下文
 */
function setupHighDPICanvas(canvas) {
  // 获取设备像素比
  const dpr = window.devicePixelRatio || 1;

  // 获取画布的显示尺寸
  let displayWidth, displayHeight;

  if (canvas.clientWidth && canvas.clientHeight) {
    // 如果canvas已经有CSS尺寸，使用它们
    displayWidth = canvas.clientWidth;
    displayHeight = canvas.clientHeight;
  } else {
    // 否则设置一个合理的默认值
    displayWidth = 800;
    displayHeight = 500;
    canvas.style.width = displayWidth + "px";
    canvas.style.height = displayHeight + "px";
  }

  // 设置画布的实际尺寸（像素）
  canvas.width = displayWidth * dpr;
  canvas.height = displayHeight * dpr;

  // 获取上下文并缩放
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  return {
    ctx,
    logicalWidth: displayWidth,
    logicalHeight: displayHeight,
    scale: dpr,
  };
}
/**
 * 绘制网格
 */
function drawGrid(ctx, width, height, padding, config) {
  ctx.strokeStyle = config.gridColor;
  ctx.lineWidth = 0.5;

  // 垂直网格线
  for (let x = padding.left; x <= width - padding.right; x += config.gridStep) {
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();
  }

  // 水平网格线
  for (
    let y = padding.top;
    y <= height - padding.bottom;
    y += config.gridStep
  ) {
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
}

/**
 * 绘制曲线
 */
function drawCurve(
  ctx,
  data,
  minValue,
  valueRange,
  chartWidth,
  chartHeight,
  padding,
  config,
) {
  const pointSpacing = chartWidth / (data.length - 1);

  ctx.strokeStyle = config.lineColor;
  ctx.lineWidth = config.lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  if (config.smooth) {
    // 绘制平滑曲线
    drawSmoothCurve(
      ctx,
      data,
      minValue,
      valueRange,
      chartWidth,
      chartHeight,
      padding,
      pointSpacing,
    );
  } else {
    // 绘制折线
    drawLinearCurve(
      ctx,
      data,
      minValue,
      valueRange,
      chartHeight,
      padding,
      pointSpacing,
    );
  }

  // 绘制数据点
  if (config.showPoints) {
    drawDataPoints(
      ctx,
      data,
      minValue,
      valueRange,
      chartHeight,
      padding,
      pointSpacing,
      config,
    );
  }
}

/**
 * 绘制平滑曲线（贝塞尔曲线）
 */
function drawSmoothCurve(
  ctx,
  data,
  minValue,
  valueRange,
  chartWidth,
  chartHeight,
  padding,
  pointSpacing,
) {
  ctx.beginPath();

  for (let i = 0; i < data.length; i++) {
    const x = padding.left + i * pointSpacing;
    const y =
      padding.top +
      chartHeight -
      ((data[i] - minValue) / valueRange) * chartHeight;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      // 使用贝塞尔曲线实现平滑
      const prevX = padding.left + (i - 1) * pointSpacing;
      const prevY =
        padding.top +
        chartHeight -
        ((data[i - 1] - minValue) / valueRange) * chartHeight;
      const cpX1 = prevX + pointSpacing * 0.5;
      const cpY1 = prevY;
      const cpX2 = x - pointSpacing * 0.5;
      const cpY2 = y;

      ctx.bezierCurveTo(cpX1, cpY1, cpX2, cpY2, x, y);
    }
  }

  ctx.stroke();
}

/**
 * 绘制折线
 */
function drawLinearCurve(
  ctx,
  data,
  minValue,
  valueRange,
  chartHeight,
  padding,
  pointSpacing,
) {
  ctx.beginPath();

  for (let i = 0; i < data.length; i++) {
    const x = padding.left + i * pointSpacing;
    const y =
      padding.top +
      chartHeight -
      ((data[i] - minValue) / valueRange) * chartHeight;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

/**
 * 绘制数据点
 */
function drawDataPoints(
  ctx,
  data,
  minValue,
  valueRange,
  chartHeight,
  padding,
  pointSpacing,
  config,
) {
  ctx.fillStyle = config.pointColor;

  for (let i = 0; i < data.length; i++) {
    const x = padding.left + i * pointSpacing;
    const y =
      padding.top +
      chartHeight -
      ((data[i] - minValue) / valueRange) * chartHeight;

    ctx.beginPath();
    ctx.arc(x, y, config.pointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * 绘制坐标轴标签（带刻度线版本）
 */
function drawAxisLabels(
  ctx,
  width,
  height,
  padding,
  minValue,
  maxValue,
  dataLength,
) {
  // 保存上下文状态
  ctx.save();

  // 设置清晰的字体
  ctx.fillStyle = "#2c3e50";
  ctx.font = '12px "Helvetica Neue", Arial, sans-serif';
  ctx.textBaseline = "middle";

  // 计算图表区域尺寸
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // 绘制坐标轴线
  drawAxisLines(ctx, width, height, padding);

  // X轴标签（数据点索引）
  ctx.textAlign = "center";
  const xLabelCount = Math.min(6, dataLength); // 最多显示6个标签
  const xStep = Math.max(1, Math.floor(dataLength / xLabelCount));

  for (let i = 0; i < dataLength; i += xStep) {
    const x = padding.left + (i / (dataLength - 1)) * chartWidth;
    const y = height - padding.bottom + 10;

    // 绘制X轴刻度线
    ctx.strokeStyle = "#2c3e50";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, height - padding.bottom - 5); // 刻度线起点
    ctx.lineTo(x, height - padding.bottom); // 刻度线终点
    ctx.stroke();

    // 添加文本阴影提高可读性
    ctx.fillStyle = "white";
    ctx.fillText(i.toString(), x, y);

    ctx.fillStyle = "#2c3e50";
    ctx.fillText(i.toString(), x, y);
  }

  // Y轴标签（数值）
  ctx.textAlign = "right";
  const yLabelCount = 5;

  for (let i = 0; i <= yLabelCount; i++) {
    const value = minValue + (maxValue - minValue) * (i / yLabelCount);
    const y = padding.top + chartHeight - (i / yLabelCount) * chartHeight;

    // 绘制Y轴刻度线
    ctx.strokeStyle = "#2c3e50";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, y); // 刻度线起点
    ctx.lineTo(padding.left + 5, y); // 刻度线终点
    ctx.stroke();

    // 格式化数值显示
    const label = formatAxisValue(value);

    // 测量文本宽度
    const textWidth = ctx.measureText(label).width;

    // 绘制标签背景（提高可读性）
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.fillRect(padding.left - textWidth - 12, y - 8, textWidth + 8, 16);

    // 绘制标签文本
    ctx.fillStyle = "#2c3e50";
    ctx.fillText(label, padding.left - 8, y);
  }

  // 恢复上下文状态
  ctx.restore();
}

/**
 * 绘制坐标轴线
 */
function drawAxisLines(ctx, width, height, padding) {
  ctx.strokeStyle = "#2c3e50";
  ctx.lineWidth = 1.5;

  // X轴线（底部）
  ctx.beginPath();
  ctx.moveTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right / 2, height - padding.bottom);
  ctx.stroke();

  // Y轴线（左侧）
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top / 2);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.stroke();

  // 添加箭头（可选）
  drawAxisArrows(ctx, width, height, padding);
}

/**
 * 格式化坐标轴数值显示
 */
function formatAxisValue(value) {
  // 根据数值大小选择合适的格式
  if (Math.abs(value) < 0.001) {
    return "0";
  } else if (Math.abs(value) < 0.01) {
    return value.toExponential(1);
  } else if (Math.abs(value) < 1) {
    return value.toFixed(3);
  } else if (Math.abs(value) < 10) {
    return value.toFixed(2);
  } else if (Math.abs(value) < 100) {
    return value.toFixed(1);
  } else {
    return value.toFixed(0);
  }
}

/**
 * 绘制坐标轴箭头（可选）
 */
function drawAxisArrows(ctx, width, height, padding) {
  let l = 8;
  // X轴箭头
  ctx.fillStyle = "#2c3e50";
  ctx.beginPath();
  ctx.moveTo(width - padding.right / 2 + l, height - padding.bottom);
  ctx.lineTo(width - padding.right / 2, height - padding.bottom - 4);
  ctx.lineTo(width - padding.right / 2, height - padding.bottom + 4);
  ctx.closePath();
  ctx.fill();

  // Y轴箭头
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top / 2 - l);
  ctx.lineTo(padding.left - 4, padding.top / 2);
  ctx.lineTo(padding.left + 4, padding.top / 2);
  ctx.closePath();
  ctx.fill();
}

// 使用示例：
/*
// HTML/CSS中需要设置canvas的显示尺寸，如
<canvas id="flowCanvas" style="width: 800px; height: 500px;"></canvas>

// JavaScript调用
const canvas = document.getElementById('flowCanvas');
// 示例数据
const sampleData = [12.5, 15.2, 14.8, 13.1, 16.7, 18.2, 17.5, 15.9, 14.3, 16.1];

// 基本使用
draw_flow_curve(canvas, sampleData);

// 自定义样式使用
draw_flow_curve(canvas, sampleData, {
    lineColor: '#e74c3c',
    lineWidth: 3,
    backgroundColor: '#2c3e50',
    gridColor: '#34495e',
    showPoints: true,
    smooth: false
});
*/
