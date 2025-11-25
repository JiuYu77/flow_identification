<template>
    <div class="address-management-container">
        <!-- 页面标题和操作栏 -->
        <div class="page-header">
            <h2>地址管理</h2>
            <div class="action-buttons">
                <el-button type="primary" @click="handleAddAddress">
                    <el-icon><Plus /></el-icon>
                    新增地址
                </el-button>
                <el-button type="danger" :disabled="selectedRows.length === 0" @click="handleBatchDelete">
                    <el-icon><Delete /></el-icon>
                    批量删除
                </el-button>
                <el-button @click="refreshAddressList">
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
                    placeholder="搜索地址名称、设备编号"
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
                
                <el-select v-model="searchParams.status" placeholder="状态筛选" clearable @change="handleSearch">
                    <el-option label="全部" :value="-1" />
                    <el-option label="启用" :value="1" />
                    <el-option label="禁用" :value="0" />
                </el-select>
            </div>
        </el-card>

        <!-- 地址列表 -->
        <el-card>
            <el-table
                :data="addressList"
                stripe
                v-loading="loading"
                style="width: 100%"
                :cell-style="{ textAlign: 'center' }"
                :header-cell-style="{ 'text-align': 'center' }"
                border
                @selection-change="handleSelectionChange"
            >
                <el-table-column type="selection" width="55" />
                <el-table-column type="index" label="序号" width="80" />
                <el-table-column prop="deviceCode" label="设备编号" min-width="120" />
                <el-table-column prop="name" label="地址名称" min-width="150" />
                <el-table-column prop="location" label="详细位置" min-width="200" />
                <el-table-column prop="deviceType" label="设备类型" width="120">
                    <template #default="scope">
                        <el-tag :type="getDeviceTypeTag(scope.row.deviceType)">
                            {{ getDeviceTypeText(scope.row.deviceType) }}
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="status" label="状态" width="100">
                    <template #default="scope">
                        <el-switch
                            v-model="scope.row.status"
                            :active-value="1"
                            :inactive-value="0"
                            @change="handleStatusChange(scope.row)"
                        />
                    </template>
                </el-table-column>
                <el-table-column label="IP地址" width="200">
                    <template #default="{ row }">
                        {{ row.ip+':'+row.port }}
                    </template>
                </el-table-column>
                <el-table-column prop="createdAt" label="创建时间" width="180">
                    <template #default="{ row }">
                        {{ formatTime(row.createdAt) }}
                    </template>
                </el-table-column>
                <el-table-column prop="updatedAt" label="更新时间" width="180">
                    <template #default="{ row }">
                        {{ formatTime(row.updatedAt) }}
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="200" fixed="right">
                    <template #default="scope">
                        <el-button type="primary" size="small" @click="handleEdit(scope.row)">编辑</el-button>
                        <el-button type="danger" size="small" @click="handleDelete(scope.row)">删除</el-button>
                    </template>
                </el-table-column>
            </el-table>

            <!-- 分页 -->
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

        <!-- 新增/编辑地址对话框 -->
        <el-dialog
            v-model="dialogVisible"
            :title="dialogTitle"
            width="500px"
        >
            <el-form :model="addressForm" :rules="rules" ref="addressFormRef" label-width="100px">
                <el-form-item label="设备编号" prop="deviceCode">
                    <el-input v-model="addressForm.deviceCode" placeholder="请输入设备编号" />
                </el-form-item>
                <el-form-item label="地址名称" prop="name">
                    <el-input v-model="addressForm.name" placeholder="请输入地址名称" />
                </el-form-item>
                <el-form-item label="详细位置" prop="location">
                    <el-input 
                        v-model="addressForm.location" 
                        type="textarea" 
                        :rows="3" 
                        placeholder="请输入详细位置描述"
                    />
                </el-form-item>
                <el-form-item label="设备类型" prop="deviceType">
                    <el-select v-model="addressForm.deviceType" placeholder="请选择设备类型" style="width: 100%">
                        <el-option label="监测设备" :value="1" />
                        <el-option label="传感器" :value="2" />
                        <el-option label="控制器" :value="3" />
                        <el-option label="其他设备" :value="4" />
                    </el-select>
                </el-form-item>
                <el-form-item label="状态" prop="status">
                    <el-radio-group v-model="addressForm.status">
                        <el-radio :label="1">启用</el-radio>
                        <el-radio :label="0">禁用</el-radio>
                    </el-radio-group>
                </el-form-item>
                <el-form-item label="IP 地址" prop="ip">
                    <el-input v-model="addressForm.ip" placeholder="请输入 IP 地址" />
                </el-form-item>
                <el-form-item label="端口号" prop="port">
                    <el-input v-model="addressForm.port" type="number" placeholder="请输入端口号" />
                </el-form-item>

                <el-form-item label="样本长度" prop="sampleLength">
                    <el-input v-model="addressForm.sampleLength" type="number" placeholder="请输入样本长度" />
                </el-form-item>
                <el-form-item label="步长" prop="step">
                    <el-input v-model="addressForm.step" type="number" placeholder="请输入步长" />
                </el-form-item>
                <el-form-item label="自动检测" prop="autoDetect">
                    <el-switch v-model="addressForm.autoDetect" :active-value="true" :inactive-value="false" />
                </el-form-item>
            </el-form>
            <template #footer>
                <el-button @click="handleDialogClose">取消</el-button>
                <el-button type="primary" @click="handleSubmit" :loading="submitting">
                    确认
                </el-button>
            </template>
        </el-dialog>
    </div>
</template>

<script setup>
// 地址管理页面逻辑
import { onMounted, ref, reactive, computed } from 'vue';
import { Search, Refresh, Plus, Delete } from '@element-plus/icons-vue';
import { formatTime } from '@/assets/js/time';
import { ElMessage, ElMessageBox } from 'element-plus';
import { addDeviceAddress, batchDeleteDeviceAddress, deleteDeviceAddress, getDeviceAddressList, updateDeviceAddress, updateDeviceAddressStatus } from '@/api/device-address';


const addressList = ref([]); // 地址数据列表
const selectedRows = ref([]); // 选中的行数据
const dialogVisible = ref(false);
const isEditMode = ref(false); // 是否为编辑模式

// 地址表单
const addressForm = reactive({
    id: 0,
    name: '',
    deviceCode: '',
    location: '',
    deviceType: 1,
    status: 1,
    ip: '',
    port: 0,
    sampleLength: 0,
    step: 0,
    autoDetect: false,
});

// 搜索参数
const searchParams = reactive({
    keyword: '',
    status: -1, // -1 表示所有状态
});

// 分页参数
const pagination = reactive({
    currentPage: 1,
    pageSize: 10,
    total: 0,
});

const loading = ref(false);
const submitting = ref(false);
const addressFormRef = ref();

// 计算属性：对话框标题
const dialogTitle = computed(() => {
    return isEditMode.value ? '编辑地址' : '新增地址';
});

// 表单验证规则
const rules = {
    name: [
        { required: true, message: '请输入地址名称', trigger: 'blur' },
        { min: 2, max: 50, message: '地址名称长度在 2 到 50 个字符', trigger: 'blur' }
    ],
    deviceCode: [
        { required: true, message: '请输入设备编号', trigger: 'blur' },
        { pattern: /^[A-Za-z0-9]+$/, message: '设备编号只能包含字母和数字', trigger: 'blur' }
    ],
    location: [
        { message: '请输入详细位置', trigger: 'blur' },
        { min: 5, max: 200, message: '位置描述长度在 5 到 200 个字符', trigger: 'blur' }
    ],
    deviceType: [
        { required: true, message: '请选择设备类型', trigger: 'change' }
    ],
    ip:[
        { required: true, message: '请输入 IP 地址', trigger: 'blur' },
        { pattern: /^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/, message: '请输入正确的 IP 地址', trigger: 'blur' }
    ],
    port:[
        { required: true, message: '请输入端口号', trigger: 'blur' },
        { 
            pattern: /^\d+$/, 
            message: '端口号必须是数字', 
            trigger: 'blur' 
        },
        { 
            validator: (rule, value, callback) => {
                if (value) {
                    const portNum = parseInt(value);
                    if (isNaN(portNum) || portNum < 0 || portNum > 65535) {
                        callback(new Error('端口号必须是 0-65535 之间的数字'));
                    } else {
                        callback();
                    }
                } else {
                    callback();
                }
            }, 
            trigger: 'blur' 
        }
    ]
};

// 设备类型标签颜色
const getDeviceTypeTag = (type) => {
    const tagMap = {
        1: '', // 传感器 - 默认
        2: 'success', // 控制器
        3: 'warning', // 监测设备
        4: 'info' // 其他设备
    };
    return tagMap[type] || '';
};

// 设备类型文本
const getDeviceTypeText = (type) => {
    const textMap = {
        1: '监测设备',
        2: '传感器',
        3: '控制器',
        4: '其他设备'
    };
    return textMap[type] || '未知';
};

// 处理选择变化
const handleSelectionChange = (selection) => {
    selectedRows.value = selection;
};

// 获取地址列表
const getAddressList = async () => {
    try {
        loading.value = true;

        const res = await getDeviceAddressList({
            keyword: searchParams.keyword,
            status: searchParams.status,
            pageNum: pagination.currentPage,
            pageSize: pagination.pageSize,
        });
        console.log('获取地址列表响应:', res);
        if (res.code == 0){
            addressList.value = res.data.device_addr_list;
            pagination.total = res.data.total;
        }
    } catch (error) {
        console.error('获取地址列表失败:', error);
        ElMessage.error('获取地址列表失败');
    } finally {
        loading.value = false;
    }
};

// 搜索
const handleSearch = () => {
    pagination.currentPage = 1;
    if(searchParams.status == undefined){
        searchParams.status = -1;
    }
    getAddressList();
};

// 刷新列表
const refreshAddressList = () => {
    searchParams.keyword = '';
    searchParams.status = -1;
    pagination.currentPage = 1;
    getAddressList();
};

// 新增地址
const handleAddAddress = () => {
    isEditMode.value = false;
    Object.assign(addressForm, {
        id: 0,
        name: '',
        deviceCode: '',
        location: '',
        deviceType: 1,
        status: 1,
        ip: '',
        port: 0,
    });
    dialogVisible.value = true;
};

// 编辑地址
const handleEdit = (address) => {
    isEditMode.value = true;
    Object.assign(addressForm, { ...address });
    dialogVisible.value = true;
};

// 删除地址
const handleDelete = async (address) => {
    try {
        await ElMessageBox.confirm(
            `确定要删除地址 "${address.deviceCode}" 吗？此操作不可恢复。`,
            '删除确认',
            {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning'
            }
        );
        
        const res = await deleteDeviceAddress({id: address.id});
        if (res.code == 0){
            ElMessage.success('地址删除成功');
            getAddressList();
        }
    } catch (error) {
        if (error.toString().includes('cancel')) {
            console.log('已取消删除操作');
        } else {
            console.error('删除地址失败:', error);
            ElMessage.error('删除地址失败');
        }
    }
};

// 批量删除
const handleBatchDelete = async () => {
    if (selectedRows.value.length === 0) {
        ElMessage.warning('请先选择要删除的地址');
        return;
    }

    try {
        const addressNames = selectedRows.value.map(address => address.name).join('、');
        await ElMessageBox.confirm(
            `确定要删除选中的 ${selectedRows.value.length} 个地址吗？\n${addressNames}\n此操作不可恢复。`,
            '批量删除确认',
            {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning',
                dangerouslyUseHTMLString: true
            }
        );
        
        const res = await batchDeleteDeviceAddress({ids: selectedRows.value.map(address => address.id)});
        if (res.code == 0){
            ElMessage.success(`成功删除 ${selectedRows.value.length} 个地址`);
            selectedRows.value = [];
            getAddressList();
        } else {
            ElMessage.error(res.msg);
        }
    } catch (error) {
        if (error.toString().includes('cancel')) {
            console.log('已取消批量删除操作');
        } else {
            console.error('批量删除失败:', error);
            ElMessage.error('批量删除失败');
        }
    }
};

// 状态切换
const handleStatusChange = async (address) => {
    try {
        const res = await updateDeviceAddressStatus({id: address.id, status: address.status});
        if (res.code == 0){
            ElMessage.success(`地址 ${address.status === 1 ? '启用' : '禁用'}成功`);
        } else {
            ElMessage.error(res.msg);
        }
    } catch (error) {
        console.error('状态更新失败:', error);
        ElMessage.error('状态更新失败');
        // 回滚状态
        address.status = address.status === 1 ? 0 : 1;
    }
};

// 提交表单
const handleSubmit = async () => {
    try {
        await addressFormRef.value.validate();
        submitting.value = true;
        let isSuccess = false;

        addressForm.port = parseInt(addressForm.port);
        addressForm.sampleLength = parseInt(addressForm.sampleLength);
        addressForm.step = parseInt(addressForm.step);

        if (isEditMode.value) {
            const res = await updateDeviceAddress(addressForm);
            if (res.code == 0){
                isSuccess = true;
                ElMessage.success('地址更新成功');
            } else {
                ElMessage.error(res.msg);
            }
        } else {
            const res = await addDeviceAddress(addressForm);
            if (res.code == 0){
                isSuccess = true;
                ElMessage.success('地址添加成功');
            } else {
                ElMessage.error(res.msg);
            }
        }

        if (isSuccess){
            dialogVisible.value = false;
            getAddressList();
        }
    } catch (error) {
        console.error('表单提交失败:', error);
    } finally {
        submitting.value = false;
    }
};

// 关闭对话框
const handleDialogClose = () => {
    dialogVisible.value = false;
    addressFormRef.value?.clearValidate();
};

// 分页大小改变
const handleSizeChange = (size) => {
    pagination.pageSize = size;
    getAddressList();
};

// 当前页改变
const handleCurrentChange = (page) => {
    pagination.currentPage = page;
    getAddressList();
};

onMounted(() => {
    getAddressList();
});
</script>

<style scoped>
.address-management-container {
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

/* 批量删除按钮样式 */
:deep(.el-button--danger:disabled) {
    opacity: 0.6;
    cursor: not-allowed;
}
</style>