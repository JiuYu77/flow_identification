<template>
    <div class="flow-analysis-container">
        <!-- 页面标题和操作栏 -->
        <div class="page-header">
            <h2>流型分析</h2>
            <div class="action-buttons">
                <el-button type="primary" @click="handleStartAnalysis">
                    <el-icon><DataAnalysis /></el-icon>
                    开始分析
                </el-button>
                <el-button type="danger" :disabled="selectedRows.length === 0" @click="handleBatchDelete">
                    <el-icon><Delete /></el-icon>
                    批量删除
                </el-button>
                <el-button @click="refreshFlowList">
                    <el-icon><Refresh /></el-icon>
                    刷新
                </el-button>
            </div>
        </div>

        <!-- 搜索和筛选 -->
        <el-card class="search-card">
            <div class="search-form">
                <el-input
                    v-model="searchParams.keyword"
                    placeholder="搜索流型名称、检测地点"
                    style="width: 300px"
                    clearable
                    @clear="handleSearch"
                    @keyup.enter="handleSearch"
                >
                    <template #append>
                        <el-button @click="handleSearch">
                            <el-icon><Search /></el-icon>
                        </el-button>
                    </template>
                </el-input>
                
                <el-select v-model="searchParams.status" placeholder="识别状态" clearable @change="handleSearch">
                    <el-option label="正确" :value="1" />
                    <el-option label="错误" :value="0" />
                </el-select>
            </div>
        </el-card>

        <el-card>
            <el-table
            :data="flowList"
            stripe
            v-loading="loading"
            style="width: 100%"
            :cell-style="{ textAlign: 'center' }"
            :header-cell-style="{ 'text-align': 'center' }"
            border
            @selection-change="handleSelectionChange"
            >
                <el-table-column type="selection" width="55" />
                <el-table-column type="index" label="序号" min-width="50" />
                <el-table-column prop="patternType" label="流型名称" />
                <el-table-column prop="predictedLabel" label="预测标签" />
                <el-table-column prop="prob" label="置信度" />
                <el-table-column prop="status" label="状态">
                    <template #default="scope">
                        <el-tag :type="scope.row.status === 1 ? 'success' : 'danger'">{{ scope.row.status === 1 ? '正确' : '错误' }}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="addr" label="检测地点" />
                <el-table-column prop="created_at" label="检测时间" min-width="150">
                    <template #default="{ row }">
                        {{ formatTime(row.created_at) }}
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="200" fixed="right">
                    <template #default="scope">
                        <el-button type="info" size="small" @click="handleChart(scope.row)">曲线</el-button>
                        <el-button type="primary" size="small" @click="handleEdit(scope.row)">修改</el-button>
                        <el-button type="danger" size="small" @click="handleDelete(scope.row)">删除</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <div class="pagination-container">
                <el-pagination
                    v-model:current-page="pagination.currentPage"
                    v-model:page-size="pagination.pageSize"
                    :page-sizes="[5, 10, 20, 50, 100]"
                    :total="pagination.total"
                    layout="total, sizes, prev, pager, next, jumper"
                    @size-change="handleSizeChange"
                    @current-change="handleCurrentChange"
                />
            </div>
        </el-card>

        <el-dialog
            v-model="dialogVisible"
            title="流型结果校正"
            width="400px"
        >
            <el-form :model="flowForm" :rules="rules" ref="flowFormRef" label-width="120px">
                <el-form-item label="流型名称" prop="patternType">
                    <el-input v-model="flowForm.patternType" />
                </el-form-item>
                <el-form-item label="预测标签" prop="predictedLabel">
                    <el-input v-model="flowForm.predictedLabel" />
                </el-form-item>
                <el-form-item label="状态" prop="status">
                    <el-radio-group v-model="flowForm.status">
                        <el-radio :label="1">正确</el-radio>
                        <el-radio :label="0">错误</el-radio>
                    </el-radio-group>
                </el-form-item>
            </el-form>
            <template #footer>
                <el-button @click="handleDialogClose">取消</el-button>
                <el-button type="primary" @click="handleSubmit" :loading="submitting">
                    确认
                </el-button>
            </template>
        </el-dialog>
        <el-dialog
            v-model="chartVisible"
            title="流型分析图表"
            width="800px"
            :style="{ padding: '20px' }"
        >
            <div class="chart-container">
                <canvas class="flowChartCanvas"></canvas>
            </div>
        </el-dialog>
    </div>
</template>

<script setup>
// 流型分析页面逻辑
import { getFlowPatternList, updateFlowPattern, deleteFlowPattern, batchDeleteFlowPattern } from '@/api/flow-pattern';
import { onMounted, ref, reactive, nextTick } from 'vue';
import { Search, Refresh, DataAnalysis, Delete } from '@element-plus/icons-vue';
import { formatTime } from '@/assets/js/time';

const flowList = ref([]); // 流型分析数据列表
const selectedRows = ref([]); // 选中的行数据
const dialogVisible = ref(false);
const flowForm = reactive({
    id: 0,
    patternType: '',
    predictedLabel: -1,
    prob: 0.0,
    status: '',
})

// 搜索参数
const searchParams = reactive({
    keyword: '',
    status: undefined, // undefined 表示 所有识别状态
})
// 分页参数
const pagination = reactive({
    currentPage: 1,
    pageSize: 10,
    total: 0,
})
const loading = ref(false);
const submitting = ref(false);

// 处理选择变化
const handleSelectionChange = (selection) => {
    selectedRows.value = selection;
}

const getFlowList = async () => {
    try{
        loading.value = true
        const res = await getFlowPatternList({
            keyword: searchParams.keyword,
            status: searchParams.status == undefined ? -1 : searchParams.status,
            page: pagination.currentPage, // 当前页码
            page_size: pagination.pageSize // 每页数量
        });
        console.log("getFlowList:", res);

        if (res.code === 0) {
            flowList.value = res.data.flow_pattern_list
            pagination.total = res.data.total
        }
    }
    catch (error) {
        console.error('获取流型分析列表失败:', error);
    }
    finally {
        loading.value = false
    }
}

onMounted(() => {
    getFlowList();
});

const refreshFlowList = () => {
    searchParams.keyword = ''
    searchParams.status = undefined
    pagination.currentPage = 1
    getFlowList()
}

const handleSearch = () => {
    console.log('searchParams:', searchParams)
    pagination.currentPage = 1
    getFlowList()
}

const handleEdit = (flow) => {
    Object.assign(flowForm, {
        id: flow.id,
        patternType: flow.patternType,
        predictedLabel: flow.predictedLabel,
        status: flow.status,
    })
    dialogVisible.value = true
}

const handleDelete = async (flow) => {
    try {
        await ElMessageBox.confirm(
            `确定要删除流型 "${flow.patternType}" 吗？此操作不可恢复。`,
            '删除确认',
            {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning'
            }
        )
        const res = await deleteFlowPattern(flow)
        if(res.code === 0){
            ElMessage.success('流型删除成功')
            getFlowList()
        }else{
            ElMessage.error('流型删除失败')
        }
    } catch (error) {
        if(error.includes('cancel') || error.toString().includes('cancel')){ // 用户取消删除
            console.log('已取消删除操作', error)
        }else{
            console.log('删除流型操作失败', error)
        }
    }
}

// 批量删除
const handleBatchDelete = async () => {
    if (selectedRows.value.length === 0) {
        ElMessage.warning('请先选择要删除的流型')
        return
    }

    try {
        const flowNames = selectedRows.value.map(flow => flow.patternType).join('、')
        await ElMessageBox.confirm(
            `确定要删除选中的 ${selectedRows.value.length} 个流型吗？\n${flowNames}\n此操作不可恢复。`,
            '批量删除确认',
            {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning',
                dangerouslyUseHTMLString: true
            }
        )
        
        const res = await batchDeleteFlowPattern({
            flow_ids: selectedRows.value.map(flow => flow.id)
        })
        if(res.code === 0){
            ElMessage.success(`成功删除 ${selectedRows.value.length} 个流型`)
            selectedRows.value = [] // 清空选择
            getFlowList()
        }else{
            ElMessage.error('批量删除失败')
        }
    } catch (error) {
        if(error.includes('cancel') || error.toString().includes('cancel')){ // 用户取消删除
            console.log('已取消批量删除操作', error)
        }else{
            console.log('批量删除操作失败', error)
        }
    }
}
const handleDialogClose = () => {
    dialogVisible.value = false
}
const handleSubmit = async () => {
    try{
        let isSuccess = false;
        submitting.value = true;

        const res = await updateFlowPattern(flowForm)
        if(res.code === 0){
            isSuccess = true
            ElMessage.success('流型数据更新成功')
        }else{
            ElMessage.error(`流型数据更新失败: ${res.msg}`)
        }
        
        if(isSuccess){
            dialogVisible.value = false; // 隐藏
            getFlowList()
        }
    }catch(e){
        console.error('流型结果校正失败:', e);
    }finally{
        submitting = false;
    }
}

const handleSizeChange = (size) => {
    pagination.pageSize = size
    getFlowList()
}
const handleCurrentChange = (page) => {
    pagination.currentPage = page
    getFlowList()
}

import { draw_flow_curve } from '@/assets/js/flow-curve'

const chartVisible = ref(false)
const handleChart = async (flow) => {
    chartVisible.value = true
    // 等待DOM更新完成，确保canvas元素已经渲染
    await nextTick();
    const canvas = document.querySelector('.flowChartCanvas')
    draw_flow_curve(canvas, flow.data)
}
</script>

<style scoped>
.flow-analysis-container {
    padding: 0;
}

.page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}
.page-header h2 {
    margin: 0;
    color: #303133;
}
.action-buttons {
    display: flex;
    gap: 12px;
}

.search-card {
    margin-bottom: 24px;
}
.search-form {
    display: flex;
    gap: 16px;
    align-items: center;
}

.pagination-container {
    margin-top: 20px;
    display: flex;
    justify-content: flex-end;
}

/* 图表容器样式 */
.chart-container {
    width: 100%;
    height: 400px;
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 4px;
    overflow: hidden;  /* 防止内容溢出 */
    border: 1px solid #81e944;
}

.flowChartCanvas {
    width: 100% !important;  /* 强制宽度为100% */
    height: 100% !important;  /* 强制高度为100% */
    display: block;  /* 确保canvas是块级元素 */
    margin: 0;
    padding: 0;
}
</style>