<template>
    <div class="data-management-container">
        <el-card>
            <template #header>
                <div class="page-header">
                    <h2>数据管理</h2>
                    <div class="header-actions">
                        <el-button type="primary" @click="handleUpload" :icon="Upload">
                            上传数据
                        </el-button>
                        <el-button @click="handleCreateDir" :icon="FolderAdd">
                            新建文件夹
                        </el-button>
                        <!-- 添加批量下载 -->
                        <el-button
                            v-if="selectedRows.length > 0"
                            @click="handleBatchDownload"
                            :icon="Download"
                            :disabled="downloadStatus.downloading"
                        >
                            批量下载 ({{ selectedRows.length }})
                        </el-button>
                        <!-- 添加批量删除 -->
                        <el-button
                            v-if="selectedRows.length > 0"
                            @click="handleBatchDelete"
                            :icon="Delete"
                            type="danger"
                            :disabled="batchDeleteLoading"
                        >
                            批量删除 ({{ selectedRows.length }})
                        </el-button>
                        <el-button @click="refreshFileList" :icon="Refresh">
                            刷新
                        </el-button>
                    </div>
                </div>
            </template>

            <div class="layout-container">
                <!-- 左侧目录树 -->
                <div class="directory-tree">
                    <div class="tree-header">
                        <span>目录结构</span>
                        <el-button 
                            size="small" 
                            type="text" 
                            @click="refreshTree"
                            :icon="Refresh"
                        >
                        </el-button>
                    </div>
                    <el-tree
                        ref="treeRef"
                        :data="directoryTree"
                        :props="treeProps"
                        node-key="path"
                        :default-expanded-keys="expandedKeys"
                        :highlight-current="true"
                        :expand-on-click-node="false"
                        @node-click="handleNodeClick"
                        class="custom-tree"
                    >
                        <template #default="{ node, data }">
                            <span class="tree-node">
                                <el-icon v-if="data.isDir" class="folder-icon">
                                    <Folder />
                                </el-icon>
                                <el-icon v-else class="file-icon">
                                    <Document />
                                </el-icon>
                                <span class="node-label">{{ node.label }}</span>
                            </span>
                        </template>
                    </el-tree>
                </div>

                <!-- 右侧文件列表 -->
                <div class="file-list-container">
                    <!-- 面包屑导航 -->
                    <div class="breadcrumb">
                        <el-breadcrumb separator="/">
                            <el-breadcrumb-item>
                                <span class="breadcrumb-link" @click="handleBreadcrumbClick('')">
                                    根目录
                                </span>
                            </el-breadcrumb-item>
                            <el-breadcrumb-item 
                                v-for="(item, index) in breadcrumbItems" 
                                :key="index"
                            >
                                <span class="breadcrumb-link" @click="handleBreadcrumbClick(item.path)">
                                    {{ item.name }}
                                </span>
                            </el-breadcrumb-item>
                        </el-breadcrumb>
                    </div>

                    <!-- 搜索区域 -->
                    <div class="search-area">
                        <el-form :model="searchParams" inline>
                            <el-form-item label="文件名">
                                <el-input 
                                    v-model="searchParams.keyword" 
                                    placeholder="请输入文件名"
                                    clearable
                                    style="width: 200px;"
                                />
                            </el-form-item>
                            <el-form-item label="文件类型">
                                <el-select 
                                    v-model="searchParams.fileType" 
                                    placeholder="请选择文件类型" 
                                    clearable
                                    filterable
                                    style="width: 200px;"
                                >
                                    <el-option label="CSV文件" value="csv" />
                                    <el-option label="Excel文件" value="excel" />
                                    <el-option label="文本文件" value="txt" />
                                    <el-option label="JSON文件" value="json" />
                                </el-select>
                            </el-form-item>
                            <el-form-item>
                                <el-button type="primary" @click="handleSearch" :icon="Search">
                                    搜索
                                </el-button>
                            </el-form-item>
                        </el-form>
                    </div>

                    <!-- 数据表格 -->
                    <div class="table-container">
                        <el-table 
                            :data="fileList" 
                            v-loading="loading"
                            @selection-change="handleSelectionChange"
                            style="width: 100%"
                        >
                            <el-table-column type="selection" width="55" />
                            <el-table-column prop="fileName" label="名称" min-width="200">
                                <template #default="{ row }">
                                    <div class="file-name-cell">
                                        <el-icon v-if="row.isDir" class="folder-icon">
                                            <Folder />
                                        </el-icon>
                                        <el-icon v-else class="file-icon">
                                            <Document />
                                        </el-icon>
                                        <span 
                                            class="file-name" 
                                            :class="{ 'dir-name': row.isDir }"
                                            @click="row.isDir ? handleDirClick(row) : null"
                                        >
                                            {{ row.fileName }}
                                        </span>
                                        <el-tag v-if="row.isDir" size="small" type="info">文件夹</el-tag>
                                    </div>
                                </template>
                            </el-table-column>
                            <el-table-column prop="fileType" label="类型" width="100">
                                <template #default="{ row }">
                                    {{ row.isDir ? '文件夹' : (row.fileType || '未知') }}
                                </template>
                            </el-table-column>
                            <el-table-column prop="fileSize" label="大小" width="120">
                                <template #default="{ row }">
                                    {{ row.isDir ? '-' : formatFileSize(row.fileSize) }}
                                </template>
                            </el-table-column>
                            <el-table-column prop="uploadTime" label="修改时间" width="180">
                                <template #default="{ row }">
                                    {{ formatTime(row.uploadTime) }}
                                </template>
                            </el-table-column>
                            <el-table-column label="操作" width="200" fixed="right">
                                <template #default="{ row }">
                                    <el-button 
                                        v-if="!row.isDir"
                                        size="small" 
                                        @click="handleDownload(row)"
                                        :icon="Download"
                                    >
                                        下载
                                    </el-button>
                                    <el-button 
                                        size="small" 
                                        type="danger" 
                                        @click="handleDelete(row)"
                                        :icon="Delete"
                                    >
                                        删除
                                    </el-button>
                                </template>
                            </el-table-column>
                        </el-table>

                        <!-- 分页 -->
                        <div class="pagination-container">
                            <el-pagination
                                v-model:current-page="pagination.currentPage"
                                v-model:page-size="pagination.pageSize"
                                :page-sizes="[10, 20, 50, 100]"
                                :total="pagination.total"
                                layout="total, sizes, prev, pager, next, jumper"
                                @size-change="handleSizeChange"
                                @current-change="handleCurrentChange"
                            />
                        </div>
                    </div>
                </div>
            </div>
        </el-card>

        <!-- 上传对话框 -->
        <el-dialog 
            v-model="uploadDialogVisible" 
            :title="`上传文件到: ${currentPath || '根目录'}`" 
            width="500px"
            :before-close="handleUploadDialogClose"
        >
            <el-upload
                class="upload-demo"
                drag
                :auto-upload="false"
                :on-change="handleFileChange"
                :show-file-list="false"
                ref="uploadRef"
            >
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">
                    将文件拖到此处，或<em>点击上传</em>
                </div>
                <template #tip>
                    <div class="el-upload__tip">
                        支持 CSV、Excel、TXT、JSON 格式文件，大小不超过 10MB
                    </div>
                </template>
            </el-upload>
            
            <!-- 已选择文件显示区域 -->
            <div v-if="selectedFile" class="selected-file-info">
                <div class="file-info-header">
                    <el-icon class="file-icon"><Document /></el-icon>
                    <span class="file-info-title">已选择文件</span>
                </div>
                <div class="file-details">
                    <div class="file-name">
                        <strong>文件名：</strong>{{ selectedFile.name }}
                    </div>
                    <div class="file-size">
                        <strong>文件大小：</strong>{{ formatFileSize(selectedFile.size) }}
                    </div>
                    <div class="file-type">
                        <strong>文件类型：</strong>{{ selectedFile.type || '未知' }}
                    </div>
                </div>
                <div class="file-actions">
                    <el-button 
                        size="small" 
                        type="danger" 
                        text 
                        @click="handleRemoveFile"
                    >
                        移除文件
                    </el-button>
                </div>
            </div>
            
            <div style="margin-top: 20px; text-align: center;">
                <el-button 
                    type="primary" 
                    @click="handleManualUpload"
                    :loading="uploadLoading"
                    :disabled="!selectedFile"
                >
                    确认上传
                </el-button>
                <el-button @click="handleUploadDialogClose">取消</el-button>
            </div>
        </el-dialog>

        <!-- 新建文件夹对话框 -->
        <el-dialog 
            v-model="createDirDialogVisible" 
            title="新建文件夹" 
            width="400px"
        >
            <el-form :model="createDirForm" label-width="80px">
                <el-form-item label="文件夹名">
                    <el-input 
                        v-model="createDirForm.name" 
                        placeholder="请输入文件夹名称"
                        clearable
                    />
                </el-form-item>
            </el-form>
            <template #footer>
                <el-button @click="createDirDialogVisible = false">取消</el-button>
                <el-button type="primary" @click="handleCreateDirConfirm">确定</el-button>
            </template>
        </el-dialog>
    </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { 
    Upload, 
    Refresh, 
    Search, 
    Download, 
    Delete, 
    Document,
    UploadFilled,
    Folder,
    FolderAdd
} from '@element-plus/icons-vue'
import { 
    getDataFileList, 
    uploadDataFile, 
    downloadDataFile, 
    deleteDataFile,
    getDirectoryTree,
    createDirectory,
    batchDeleteDataFile
} from '@/api/flow-data'
import { formatTime } from '@/assets/js/time'

// 响应式数据
const loading = ref(false)
const uploadDialogVisible = ref(false)
const createDirDialogVisible = ref(false)
const uploadLoading = ref(false)
const selectedFile = ref(null)
const fileList = ref([])
const selectedRows = ref([])
const directoryTree = ref([])
const currentPath = ref('')
const expandedKeys = ref([''])
const uploadRef = ref(null)
const batchDeleteLoading = ref(false)  // 批量删除加载状态


// 树形配置
const treeProps = {
    children: 'children',
    label: 'fileName'
}

// 新建文件夹表单
const createDirForm = reactive({
    name: ''
})

// 搜索参数
const searchParams = reactive({
    keyword: '',
    fileType: ''
})

// 分页参数
const pagination = reactive({
    currentPage: 1,
    pageSize: 10,
    total: 0
})


// 面包屑导航项
const breadcrumbItems = computed(() => {
    if (!currentPath.value) return []
    return currentPath.value.split('/').map((segment, index, array) => ({
        name: segment,
        path: array.slice(0, index + 1).join('/')
    }))
})

// 获取目录树
const getDirectoryTreeData = async () => {
    try {
        const res = await getDirectoryTree()
        if (res.code === 0) {
            directoryTree.value = res.data.tree || []
        }
    } catch (error) {
        console.error('获取目录树失败:', error)
    }
}

// 获取文件列表
const getFileList = async () => {
    try {
        loading.value = true
        const res = await getDataFileList({
            keyword: searchParams.keyword,
            fileType: searchParams.fileType,
            path: currentPath.value,
            page: pagination.currentPage,
            pageSize: pagination.pageSize
        })

        if (res.code === 0) {
            fileList.value = res.data.fileList || []
            pagination.total = res.data.total || 0
        } else {
            ElMessage.error(res.msg || '获取文件列表失败')
        }
    } catch (error) {
        console.error('获取文件列表失败:', error)
        ElMessage.error('获取文件列表失败')
    } finally {
        loading.value = false
    }
}

// 刷新目录树
const refreshTree = () => {
    getDirectoryTreeData()
}

// 树节点点击
const handleNodeClick = (data) => {
    currentPath.value = data.path ? `${data.path}/${data.fileName}` : data.fileName
    getFileList()
}

// 面包屑点击
const handleBreadcrumbClick = (path) => {
    currentPath.value = path
    getFileList()
}

// 文件夹点击
const handleDirClick = (row) => {
    const newPath = currentPath.value ? `${currentPath.value}/${row.fileName}` : row.fileName
    currentPath.value = newPath
    getFileList()
}

// 搜索
const handleSearch = () => {
    pagination.currentPage = 1
    getFileList()
}

// 刷新
const refreshFileList = () => {
    searchParams.keyword = ''
    searchParams.fileType = ''
    currentPath.value = ''
    pagination.currentPage = 1
    getFileList()
}

// 上传文件
const handleUpload = () => {
    uploadDialogVisible.value = true
}

// 新建文件夹
const handleCreateDir = () => {
    createDirForm.name = ''
    createDirDialogVisible.value = true
}

// 确认新建文件夹
const handleCreateDirConfirm = async () => {
    if (!createDirForm.name.trim()) {
        ElMessage.warning('请输入文件夹名称')
        return
    }

    try {
        const res = await createDirectory({
            path: currentPath.value,
            name: createDirForm.name
        })

        if (res.code === 0) {
            ElMessage.success('文件夹创建成功')
            createDirDialogVisible.value = false
            getFileList()
            getDirectoryTreeData()
        } else {
            ElMessage.error(res.msg || '创建文件夹失败')
        }
    } catch (error) {
        console.error('创建文件夹失败:', error)
        ElMessage.error('创建文件夹失败')
    }
}

// 删除文件/文件夹
const handleDelete = async (row) => {
    try {
        const message = row.isDir ? 
            `确定要删除文件夹 "${row.fileName}" 及其所有内容吗？` :
            `确定要删除文件 "${row.fileName}" 吗？`
        
        await ElMessageBox.confirm(
            message,
            '删除确认',
            {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning'
            }
        )
        
        const res = await deleteDataFile({
            path: currentPath.value,
            name: row.fileName,
            isDir: row.isDir
        })

        if (res.code === 0) {
            ElMessage.success('删除成功')
            getFileList()
            getDirectoryTreeData()
        } else {
            ElMessage.error(res.msg || '删除失败')
        }
    } catch (error) {
        if (error !== 'cancel') {
            console.error('删除失败:', error)
            ElMessage.error('删除失败')
        }
    }
}

// 在脚本部分添加下载状态管理
const downloadStatus = reactive({
    downloading: false,
    currentFile: '',
    progress: 0
})

// 下载文件方法
const handleDownload = async (row) => {
    if (downloadStatus.downloading) {
        ElMessage.warning('当前有文件正在下载，请稍后')
        return
    }
    
    try {
        downloadStatus.downloading = true
        downloadStatus.currentFile = row.fileName
        downloadStatus.progress = 0
        
        // 显示下载提示
        ElMessage.info(`开始下载: ${row.fileName}`)
        
        await downloadDataFile(row.fileName, currentPath.value)
        
        ElMessage.success(`文件下载成功: ${row.fileName}`)
    } catch (error) {
        console.error('下载文件失败:', error)
        ElMessage.error(`下载失败: ${error.message}`)
    } finally {
        downloadStatus.downloading = false
        downloadStatus.currentFile = ''
        downloadStatus.progress = 0
    }
}

// 批量下载
const handleBatchDownload = async () => {
    if (selectedRows.value.length === 0) {
        ElMessage.warning('请选择要下载的文件')
        return
    }
    
    try {
        ElMessage.info(`开始批量下载 ${selectedRows.value.length} 个文件`)
        
        for (const file of selectedRows.value) {
            if (!file.isDir) {
                await handleDownload(file)
            }
        }
        
        ElMessage.success('批量下载完成')
    } catch (error) {
        console.error('批量下载失败:', error)
    }
}
// 批量删除文件
const handleBatchDelete = async () => {
    if (selectedRows.value.length === 0) {
        ElMessage.warning('请选择要删除的文件')
        return
    }

    try {
        // 确认删除对话框
        await ElMessageBox.confirm(
            `确定要删除选中的 ${selectedRows.value.length} 个文件/文件夹吗？此操作不可恢复。`,
            '批量删除确认',
            {
                confirmButtonText: '确定删除',
                cancelButtonText: '取消',
                type: 'warning',
                dangerouslyUseHTMLString: true
            }
        )
        
        batchDeleteLoading.value = true
        
        // 准备批量删除数据
        const deleteItems = selectedRows.value.map(item => ({
            fileName: item.fileName,
            path: currentPath.value,
            isDir: item.isDir || false
        }))
        
        const res = await batchDeleteDataFile(deleteItems)
        
        if (res.code === 0) {
            ElMessage.success(`成功删除 ${selectedRows.value.length} 个文件/文件夹`)
            // 清空选中项
            selectedRows.value = []
            // 刷新文件列表和目录树
            getFileList()
            getDirectoryTreeData()
        } else {
            ElMessage.error(res.msg || '批量删除失败')
        }
    } catch (error) {
        if (error !== 'cancel') {
            console.error('批量删除失败:', error)
            ElMessage.error('批量删除失败')
        }
    } finally {
        batchDeleteLoading.value = false
    }
}

// 文件选择变化
const handleFileChange = (file) => {
    selectedFile.value = file.raw
}

// 移除已选择文件
const handleRemoveFile = () => {
    selectedFile.value = null
    if (uploadRef.value) {
        uploadRef.value.clearFiles()
    }
}

// 关闭上传对话框
const handleUploadDialogClose = () => {
    uploadDialogVisible.value = false
    selectedFile.value = null
    uploadLoading.value = false
    if (uploadRef.value) {
        uploadRef.value.clearFiles()
    }
}

// 手动上传文件
const handleManualUpload = async () => {
    if (!selectedFile.value) {
        ElMessage.warning('请选择要上传的文件')
        return
    }

    // 验证文件 - 修复：直接传递selectedFile.value
    if (!beforeUpload(selectedFile.value)) {
        return
    }

    uploadLoading.value = true
    try {
        const formData = new FormData()
        // 修复：直接使用selectedFile.value而不是selectedFile.value.raw
        formData.append('file', selectedFile.value)
        formData.append('path', currentPath.value)
        formData.append('category', 'flow-data')

        const res = await uploadDataFile(formData)

        if (res.code === 0) {
            ElMessage.success('文件上传成功')
            uploadDialogVisible.value = false
            selectedFile.value = null
            if (uploadRef.value) {
                uploadRef.value.clearFiles()
            }
            getFileList()
            getDirectoryTreeData()
        } else {
            ElMessage.error(res.msg || '文件上传失败')
        }
    } catch (error) {
        console.error('文件上传失败:', error)
        ElMessage.error('文件上传失败')
    } finally {
        uploadLoading.value = false
    }
}

// 上传前验证
const beforeUpload = (file) => {
    const allowedTypes = [
        'text/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'text/plain',
        'application/json'
    ]
    const isLt10M = file.size / 1024 / 1024 < 10

    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(csv|xlsx?|txt|json)$/)) {
        ElMessage.error('只能上传 CSV、Excel、TXT、JSON 格式文件!')
        return false
    }
    if (!isLt10M) {
        ElMessage.error('文件大小不能超过 10MB!')
        return false
    }
    return true
}

// 选择变化
const handleSelectionChange = (selection) => {
    selectedRows.value = selection
}

// 分页大小变化
const handleSizeChange = (size) => {
    pagination.pageSize = size
    pagination.currentPage = 1
    getFileList()
}

// 页码变化
const handleCurrentChange = (page) => {
    pagination.currentPage = page
    getFileList()
}

// 格式化文件大小
const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

onMounted(() => {
    getFileList()
    getDirectoryTreeData()
})
</script>

<style scoped>
.data-management-container {
    padding: 0;
}

.page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.page-header h2 {
    margin: 0;
    color: #303133;
}

.header-actions {
    display: flex;
    gap: 12px;
}

.layout-container {
    display: flex;
    gap: 20px;
    min-height: 600px;
}

.directory-tree {
    width: 280px;
    border-right: 1px solid #e4e7ed;
    padding-right: 20px;
}

.tree-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
    font-weight: 600;
    color: #303133;
}

.custom-tree {
    max-height: 500px;
    overflow-y: auto;
}

.tree-node {
    display: flex;
    align-items: center;
    gap: 8px;
}

.folder-icon {
    color: #e6a23c;
}

.file-icon {
    color: #409eff;
}

.node-label {
    font-size: 14px;
}

.file-list-container {
    flex: 1;
    min-width: 0;
}

.breadcrumb {
    margin-bottom: 20px;
    padding: 12px 16px;
    background: #f5f7fa;
    border-radius: 4px;
}

.breadcrumb-link {
    color: #606266;
    cursor: pointer;
    text-decoration: none;
}

.breadcrumb-link:hover {
    color: #409eff;
    text-decoration: underline;
}

.search-area {
    margin-bottom: 20px;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
}

.table-container {
    margin-top: 20px;
}

.file-name-cell {
    display: flex;
    align-items: center;
    gap: 8px;
}

.file-name {
    font-weight: 500;
    cursor: default;
}

.dir-name {
    color: #409eff;
    cursor: pointer;
    text-decoration: underline;
}

.dir-name:hover {
    color: #66b1ff;
}

.pagination-container {
    margin-top: 20px;
    display: flex;
    justify-content: flex-end;
}

/* 已选择文件信息样式 */
.selected-file-info {
    margin-top: 20px;
    padding: 16px;
    border: 1px solid #e4e7ed;
    border-radius: 8px;
    background-color: #f8f9fa;
}

.file-info-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    font-weight: 600;
    color: #303133;
}

.file-info-title {
    margin-left: 8px;
}

.file-details {
    margin-bottom: 12px;
}

.file-details div {
    margin-bottom: 4px;
    font-size: 14px;
    color: #606266;
}

.file-actions {
    text-align: right;
}

/* 上传区域样式优化 */
.upload-demo {
    margin-bottom: 20px;
}
</style>