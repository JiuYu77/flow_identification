<template>
    <div class="user-management-container">
        <!-- 页面标题和操作栏 -->
        <div class="page-header">
            <h2>用户管理</h2>
            <div class="action-buttons">
                <el-button type="primary" @click="handleAddUser">
                    <el-icon><Plus /></el-icon>
                    添加用户
                </el-button>
                <el-button @click="refreshUserList">
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
                    placeholder="搜索用户名、邮箱或昵称"
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
                
                <el-select v-model="searchParams.identity" placeholder="用户身份" clearable @change="handleSearch">
                    <el-option label="普通用户" :value="2" />
                    <el-option label="管理员" :value="1" />
                    <el-option label="所有用户" :value="-1" />
                </el-select>
            </div>
        </el-card>

        <!-- 用户列表表格 -->
        <el-card>
            <el-table
                :data="userList"
                v-loading="loading"
                stripe
                style="width: 100%"
            >
                <el-table-column type="index" label="序号" width="60" />
                <el-table-column prop="username" label="用户名" min-width="120" />
                <el-table-column prop="email" label="邮箱" min-width="180" />
                <el-table-column prop="nickname" label="昵称" min-width="120" />
                <el-table-column prop="identity" label="身份" width="100">
                    <template #default="scope">
                        <el-tag :type="scope.row.identity === 1 ? 'success' : 'info'">
                            {{ scope.row.identity === 1 ? '管理员' : '普通用户' }}
                        </el-tag>
                    </template>
                </el-table-column>
                <el-table-column prop="created_at" label="创建时间" width="180">
                    <template #default="{ row }">
                        {{ formatTime(row.created_at) }}
                    </template>
                </el-table-column>
                <el-table-column label="操作" width="200" fixed="right">
                    <template #default="scope">
                        <el-button
                            size="small"
                            type="primary"
                            @click="handleEdit(scope.row)"
                        >
                            编辑
                        </el-button>
                        <el-button
                            size="small"
                            type="danger"
                            @click="handleDelete(scope.row)"
                            v-if="scope.row.id !== currentUser.id"
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
                    :page-sizes="[5, 10, 20, 50, 100]"
                    :total="pagination.total"
                    layout="total, sizes, prev, pager, next, jumper"
                    @size-change="handleSizeChange"
                    @current-change="handleCurrentChange"
                />
            </div>
        </el-card>

        <!-- 添加/编辑用户对话框 -->
        <el-dialog
            v-model="dialogVisible"
            :title="dialogTitle"
            width="500px"
            :before-close="handleDialogClose"
        >
            <el-form
                ref="userFormRef"
                :model="userForm"
                :rules="userFormRules"
                label-width="80px"
            >
                <el-form-item label="用户名" prop="username">
                    <el-input v-model="userForm.username" placeholder="请输入用户名" />
                </el-form-item>
                <el-form-item label="邮箱" prop="email">
                    <el-input v-model="userForm.email" placeholder="请输入邮箱" />
                </el-form-item>
                <el-form-item label="昵称" prop="nickname">
                    <el-input v-model="userForm.nickname" placeholder="请输入昵称" />
                </el-form-item>
                <el-form-item label="密码" prop="password" v-if="isAddMode">
                    <el-input
                        v-model="userForm.password"
                        type="password"
                        placeholder="请输入密码"
                        show-password
                    />
                </el-form-item>
                <el-form-item label="确认密码" prop="confirm_password" v-if="isAddMode">
                    <el-input
                        v-model="userForm.confirm_password"
                        type="password"
                        placeholder="请输入密码"
                        show-password
                    />
                </el-form-item>
                <el-form-item label="身份" prop="identity">
                    <el-radio-group v-model="userForm.identity">
                        <el-radio :label="2">普通用户</el-radio>
                        <el-radio :label="1">管理员</el-radio>
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
    </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useUserStore } from '@/store/user.js'
import { register, updateUser, deleteUser, fetchUserList } from '@/api/user.js'
import { Plus, Refresh, Search } from '@element-plus/icons-vue'
import { formatTime } from '@/assets/js/time';


const userStore = useUserStore()
const loading = ref(false)
const dialogVisible = ref(false)
const submitting = ref(false)
const userFormRef = ref()

// 当前用户信息
const currentUser = computed(() => ({
    id: userStore.id,
    username: userStore.username,
    email: userStore.email,
    nickname: userStore.nickname,
    identity: userStore.identity
}))

// 搜索参数
const searchParams = reactive({
    keyword: '',
    identity: -1, // -1 表示查询所有身份,  1 表示管理员, 2 表示普通用户
})

// 分页参数
const pagination = reactive({
    currentPage: 1,
    pageSize: 10,
    total: 0,
})

// 用户列表数据
const userList = ref([])

// 用户表单
const userForm = reactive({
    id: '',
    username: '',
    email: '',
    nickname: '',
    password: '',
    confirm_password: '',
    identity: 2
})

// 表单验证规则
const userFormRules = {
    username: [
        { required: true, message: '请输入用户名', trigger: 'blur' },
        { min: 3, max: 20, message: '用户名长度在 3 到 20 个字符', trigger: 'blur' }
    ],
    email: [
        { required: true, message: '请输入邮箱地址', trigger: 'blur' },
        { type: 'email', message: '请输入正确的邮箱地址', trigger: 'blur' }
    ],
    password: [
        { required: true, message: '请输入密码', trigger: 'blur' },
        { min: 8, message: '密码长度不能少于 8 个字符', trigger: 'blur' }
    ],
    confirm_password: [
        { required: true, message: '请再次输入密码', trigger: 'blur' },
        { 
            validator: (rule, value, callback) => {
                if (value !== userForm.password) {
                    callback(new Error('两次输入密码不一致!'));
                } else {
                    callback();
                }
            }, 
            trigger: 'blur' 
      }
    ]
}

// 计算属性
const isAddMode = computed(() => !userForm.id)
const dialogTitle = computed(() => isAddMode.value ? '添加用户' : '编辑用户')

// 方法
const getUserList = async () => {
    loading.value = true
    try {
        const res = await fetchUserList({
            keyword: searchParams.keyword,
            identity: searchParams.identity,
            page: pagination.currentPage, // 当前页码
            page_size: pagination.pageSize // 每页数量
        });

        console.log("getUserList:", res)

        if(res.code == 0){
            userList.value = res.data.user_list
            pagination.total = res.data.total
        }
    } catch (error) {
        ElMessage.error('获取用户列表失败')
    } finally {
        loading.value = false
    }
}

const handleSearch = () => {
    pagination.currentPage = 1
    if(searchParams.identity == undefined){ // 清空 el-select 后，重置
        searchParams.identity = -1;
    }
    getUserList()
}

const refreshUserList = () => {
    searchParams.keyword = ''
    searchParams.identity = -1
    pagination.currentPage = 1
    getUserList()
}

const handleAddUser = () => {
    Object.assign(userForm, {
        id: '',
        username: '',
        email: '',
        nickname: '',
        password: '',
        confirm_password: '',
        identity: 2
    })
    dialogVisible.value = true
}

const handleEdit = (user) => {
    Object.assign(userForm, {
        id: user.id,
        username: user.username,
        email: user.email,
        nickname: user.nickname,
        password: '',
        confirm_password: '',
        identity: user.identity
    })
    dialogVisible.value = true
}

const handleDelete = async (user) => {
    try {
        await ElMessageBox.confirm(
            `确定要删除用户 "${user.username}" 吗？此操作不可恢复。`,
            '删除确认',
            {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning'
            }
        )
        const res = await deleteUser(user)
        if(res.code === 0){
            ElMessage.success('用户删除成功')
            getUserList()
        }else{
            ElMessage.error('用户删除失败')
        }
    } catch (error) {
        if(error.includes('cancel') || error.toString().includes('cancel')){ // 用户取消删除
            console.log('已取消删除操作', error)
        }else{
            console.log('删除用户操作失败', error)
        }
    }
}


const handleSubmit = async () => {
    if (!userFormRef.value) return
    
    try {
        await userFormRef.value.validate()
        submitting.value = true
        
        let isSuccess = false;

        if (isAddMode.value) {
            const res = await register(userForm);
            if(res.code === 0){
                isSuccess = true;
                ElMessage.success('用户添加成功')
            }else{
                ElMessage.error(res.msg || '用户添加失败')
            }
        } else {
            const res = await updateUser(userForm);
            if(res.code === 0){
                isSuccess = true;
                ElMessage.success('用户信息更新成功')
            }else{
                console.log(res.msg);
                ElMessage.error('用户信息更新失败')
            }
        }
        
        if(isSuccess){
            dialogVisible.value = false; // 隐藏
            getUserList()
        }
    } catch (error) {
        // 表单验证失败
    } finally {
        submitting.value = false
    }
}

const handleDialogClose = () => {
    dialogVisible.value = false
    userFormRef.value?.resetFields()
}

const handleSizeChange = (size) => {
    pagination.pageSize = size
    getUserList()
}

const handleCurrentChange = (page) => {
    pagination.currentPage = page
    getUserList()
}

// 生命周期
onMounted(() => {
    getUserList()
})
</script>

<style scoped>
.user-management-container {
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
</style>