<template>
    <div class="user-profile-container">
        <!-- 页面标题 -->
        <div class="page-header">
            <h2>个人信息</h2>
            <p>查看和编辑您的个人资料</p>
        </div>

        <!-- 主要内容区域 -->
        <div class="profile-content">
            <!-- 左侧：头像和基本信息 -->
            <div class="profile-left">
                <el-card class="avatar-card">
                    <div class="avatar-section">
                        <el-avatar 
                            :size="120" 
                            :src="userStore.avatar" 
                            class="user-avatar"
                        >
                            {{ userStore.nickname?.charAt(0) || userStore.username?.charAt(0) || 'U' }}
                        </el-avatar>
                        <div class="avatar-actions">
                            <el-button type="primary" size="small" @click="showAvatarDialog">
                                <el-icon><Upload /></el-icon>
                                更换头像
                            </el-button>
                            <el-button size="small" @click="resetAvatar">
                                <el-icon><Refresh /></el-icon>
                                重置头像
                            </el-button>
                        </div>
                    </div>

                    <div class="basic-info">
                        <h3>{{ userStore.nickname || userStore.username || '未设置昵称' }}</h3>
                        <p class="username">@{{ userStore.username }}</p>
                        <p class="identity">
                            <el-tag :type="userStore.identity === 1 ? 'success' : 'info'">
                                {{ userStore.identity === 1 ? '管理员' : '普通用户' }}
                            </el-tag>
                        </p>
                        <p class="join-time">注册时间：{{ formatTime(userStore.created_at) }}</p>
                    </div>
                </el-card>
            </div>

            <!-- 右侧：详细信息表单 -->
            <div class="profile-right">
                <el-card class="info-card">
                    <template #header>
                        <div class="card-header">
                            <span>基本信息</span>
                            <div class="action-buttons">
                                <el-button 
                                    v-if="isEditMode"
                                    size="small" 
                                    @click="cancelEdit"
                                    :icon="Close"
                                >
                                    取消
                                </el-button>
                                <el-button 
                                    type="primary" 
                                    size="small" 
                                    @click="toggleEditMode"
                                    :icon="isEditMode ? Check : Edit"
                                >
                                    {{ isEditMode ? '保存' : '编辑' }}
                                </el-button>
                            </div>
                        </div>
                    </template>

                    <el-form 
                        ref="userFormRef" 
                        :model="userStore" 
                        :rules="formRules"
                        label-width="100px"
                        :disabled="!isEditMode"
                    >
                        <el-form-item label="用户ID" prop="id">
                            <el-input v-model="userStore.id" disabled />
                        </el-form-item>

                        <el-form-item label="用户名" prop="username">
                            <el-input v-model="userStore.username" />
                        </el-form-item>

                        <el-form-item label="邮箱" prop="email">
                            <el-input v-model="userStore.email" type="email" />
                        </el-form-item>

                        <el-form-item label="昵称" prop="nickname">
                            <el-input v-model="userStore.nickname" placeholder="请输入昵称" />
                        </el-form-item>

                        <el-form-item label="身份" prop="identity">
                            <el-select v-model="userStore.identity" disabled :suffix-icon="null">
                                <el-option :value="2" label="普通用户" />
                                <el-option :value="1" label="管理员" />
                            </el-select>
                        </el-form-item>

                        <el-form-item label="创建时间" prop="created_at">
                            <el-input v-model="userStore.created_at" disabled />
                        </el-form-item>

                        <el-form-item label="更新时间" prop="updated_at">
                            <el-input v-model="userStore.updated_at" disabled />
                        </el-form-item>
                    </el-form>
                </el-card>

                <!-- 密码修改卡片 -->
                <el-card class="password-card">
                    <template #header>
                        <div class="card-header">
                            <span>修改密码</span>
                            <el-button 
                                type="warning" 
                                size="small" 
                                @click="showPasswordDialog"
                                :icon="EditPen"
                            >
                                修改密码
                            </el-button>
                        </div>
                    </template>

                    <div class="password-info">
                        <p>为了账户安全，建议定期修改密码</p>
                        <p class="last-update">最后修改时间：{{ formatTime(userStore.updated_at) }}</p>
                    </div>
                </el-card>
            </div>
        </div>

        <!-- 头像上传对话框 -->
        <el-dialog 
            v-model="avatarDialogVisible" 
            title="更换头像" 
            width="400px"
            :before-close="handleAvatarDialogClose"
        >
            <div class="avatar-upload">
                <el-upload
                    class="avatar-uploader"
                    action="#"
                    :show-file-list="false"
                    :before-upload="beforeAvatarUpload"
                    :http-request="handleAvatarUpload"
                >
                    <el-avatar v-if="tempAvatar" :size="100" :src="tempAvatar" />
                    <el-icon v-else class="avatar-uploader-icon"><Plus /></el-icon>
                </el-upload>
                <div class="upload-tips">
                    <p>支持 JPG、PNG 格式，大小不超过 2MB</p>
                    <p>建议尺寸：200x200 像素</p>
                </div>
            </div>
            <template #footer>
                <el-button @click="handleAvatarDialogClose">取消</el-button>
                <el-button type="primary" @click="confirmAvatarChange" :loading="avatarUploading">
                    确认更换
                </el-button>
            </template>
        </el-dialog>

        <!-- 密码修改对话框 -->
        <el-dialog 
            v-model="passwordDialogVisible" 
            title="修改密码" 
            width="500px"
            :before-close="handlePasswordDialogClose"
        >
            <el-form 
                ref="passwordFormRef" 
                :model="passwordForm" 
                :rules="passwordRules"
                label-width="120px"
            >
                <el-form-item label="当前密码" prop="current_password">
                    <el-input 
                        v-model="passwordForm.current_password" 
                        type="password" 
                        show-password 
                        placeholder="请输入当前密码"
                    />
                </el-form-item>

                <el-form-item label="新密码" prop="new_password">
                    <el-input 
                        v-model="passwordForm.new_password" 
                        type="password" 
                        show-password 
                        placeholder="请输入新密码"
                    />
                </el-form-item>

                <el-form-item label="确认新密码" prop="confirm_password">
                    <el-input 
                        v-model="passwordForm.confirm_password" 
                        type="password" 
                        show-password 
                        placeholder="请再次输入新密码"
                    />
                </el-form-item>
            </el-form>
            <template #footer>
                <el-button @click="handlePasswordDialogClose">取消</el-button>
                <el-button 
                    type="primary" 
                    @click="confirmPasswordChange" 
                    :loading="passwordChanging"
                >
                    确认修改
                </el-button>
            </template>
        </el-dialog>
    </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useUserStore } from '@/store/user'
import { getUserProfile, updateUserProfile, changePassword, updateUserAvatar } from '@/api/user'
import { 
    Upload, 
    Refresh, 
    Check, 
    Edit, 
    EditPen, 
    Plus,
    Close
} from '@element-plus/icons-vue'
import { formatTime } from '@/assets/js/time';


// 用户状态管理
const userStore = useUserStore()

// 响应式数据
const isEditMode = ref(false)
const avatarDialogVisible = ref(false)
const passwordDialogVisible = ref(false)
const avatarUploading = ref(false)
const passwordChanging = ref(false)
const tempAvatar = ref('')
const userFormRef = ref()
const passwordFormRef = ref()


// 密码表单数据
const passwordForm = reactive({
    current_password: '',
    new_password: '',
    confirm_password: ''
})

// 表单验证规则
const formRules = {
    username: [
        { required: true, message: '请输入用户名', trigger: 'blur' },
        { min: 3, max: 20, message: '用户名长度在 3 到 20 个字符', trigger: 'blur' }
    ],
    email: [
        { required: true, message: '请输入邮箱地址', trigger: 'blur' },
        { type: 'email', message: '请输入正确的邮箱地址', trigger: 'blur' }
    ],
    nickname: [
        { max: 20, message: '昵称长度不能超过 20 个字符', trigger: 'blur' }
    ]
}

// 密码验证规则
const passwordRules = {
    current_password: [
        { required: true, message: '请输入当前密码', trigger: 'blur' }
    ],
    new_password: [
        { required: true, message: '请输入新密码', trigger: 'blur' },
        { min: 8, message: '密码长度不能少于 8 个字符', trigger: 'blur' }
    ],
    confirm_password: [
        { required: true, message: '请确认新密码', trigger: 'blur' },
        { 
            validator: (rule, value, callback) => {
                if (value !== passwordForm.new_password) {
                    callback(new Error('两次输入密码不一致'))
                } else {
                    callback()
                }
            }, 
            trigger: 'blur' 
        }
    ]
}

// 获取用户信息
const fetchUserProfile = async () => {
    try {
        const res = await getUserProfile()
        if (res.code === 0) {
            userStore.$patch(res.data)
        } else {
            ElMessage.error('获取用户信息失败')
        }
    } catch (error) {
        ElMessage.error('获取用户信息失败')
    }
}

// 切换编辑模式
const toggleEditMode = async () => {
    if (isEditMode.value) {
        // 保存模式
        try {
            await userFormRef.value.validate()
            const res = await updateUserProfile(userStore.gets)
            if (res.code === 0) {
                ElMessage.success('个人信息更新成功')
                fetchUserProfile(); // 更新用户状态
                isEditMode.value = false;
            } else {
                ElMessage.error(res.msg || '更新失败')
            }
        } catch (error) {
            console.log("表单验证失败:", userStore)
            ElMessage.error('表单验证失败')
        }
    } else {
        // 编辑模式
        isEditMode.value = true
    }
}

// 取消编辑
const cancelEdit = () => {
    // 重置表单数据为原始数据
    fetchUserProfile()
    isEditMode.value = false
    ElMessage.info('已取消编辑')
}

// 显示密码修改对话框
const showPasswordDialog = () => {
    Object.assign(passwordForm, {
        current_password: '',
        new_password: '',
        confirm_password: ''
    })
    passwordDialogVisible.value = true
}
// 确认修改密码
const confirmPasswordChange = async () => {
    try {
        await passwordFormRef.value.validate()
        passwordChanging.value = true
        
        const res = await changePassword(passwordForm)
        if (res.code === 0) {
            ElMessage.success('密码修改成功')
            passwordDialogVisible.value = false
            // 清空表单
            Object.assign(passwordForm, {
                current_password: '',
                new_password: '',
                confirm_password: ''
            })
            fetchUserProfile();
        } else {
            ElMessage.error(res.msg || '密码修改失败')
        }
    } catch (error) {
        ElMessage.error('表单验证失败')
    } finally {
        passwordChanging.value = false
    }
}

// 显示头像上传对话框
const showAvatarDialog = () => {
    tempAvatar.value = userStore.avatar
    avatarDialogVisible.value = true
}

// 重置头像
const resetAvatar = async () => {
    try{
        // 显示确认对话框
        await ElMessageBox.confirm(
            '确定要重置头像吗？重置后将显示默认字母头像。',
            '重置头像确认',
            {
                confirmButtonText: '确定重置',
                cancelButtonText: '取消',
                type: 'warning',
                center: true
            }
        )
        const formData = new FormData()
        formData.append('reset_avatar', true)

        // 用户确认后执行重置
        const res = await updateUserAvatar(formData)
        if (res.code === 0) {
            userStore.avatar = ''
            ElMessage.success('已重置为默认头像')
        }
    }catch(e){

    }
}

// 头像上传前验证
const beforeAvatarUpload = (file) => {
    const isJPGOrPNG = file.type === 'image/jpeg' || file.type === 'image/png'
    const isLt2M = file.size / 1024 / 1024 < 2

    if (!isJPGOrPNG) {
        ElMessage.error('头像只能是 JPG 或 PNG 格式!')
        return false
    }
    if (!isLt2M) {
        ElMessage.error('头像大小不能超过 2MB!')
        return false
    }
    return true
}

const selectedFile = ref(null)

// 打开选择的头像
const handleAvatarUpload = async (options) => {
    const { file } = options;
    try{
        // 只创建预览，不上传到服务器
        const reader = new FileReader()
        reader.onload = (e) => {
            tempAvatar.value = e.target.result
            ElMessage.success('头像预览成功，请点击确认更换按钮完成上传')
        }
        reader.readAsDataURL(file)
        
        // 保存文件对象，用于后续上传
        selectedFile.value = file;
    } catch (error) {
        ElMessage.error('头像预览失败')
    }
}

// 确认更换头像
const confirmAvatarChange = async () => {
    avatarUploading.value = true
    
    try {
        const formData = new FormData()
        formData.append('avatar', selectedFile.value)
        formData.append('reset_avatar', false)

        const res = await updateUserAvatar(formData)
        console.log("confirmAvatarChange...:", res)
        if (res.code === 0) {
            fetchUserProfile();
            ElMessage.success('头像更换成功')
            avatarDialogVisible.value = false;
        } else {
            ElMessage.error(res.msg || '头像上传失败')
        }
    } catch (error) {
        ElMessage.error('头像上传失败')
    } finally {
        avatarUploading.value = false
        selectedFile.value = null
    }
}

// 对话框关闭处理
const handleAvatarDialogClose = () => {
    avatarDialogVisible.value = false
    tempAvatar.value = ''
    selectedFile.value = null
}

const handlePasswordDialogClose = () => {
    passwordDialogVisible.value = false
    passwordFormRef.value?.resetFields()
}

// 组件挂载时获取用户信息
onMounted(() => {
    fetchUserProfile()
})
</script>

<style scoped>
.user-profile-container {
    padding: 0;
    max-width: 1200px;
    margin: 0 auto;
}

.page-header {
    margin-bottom: 32px;
    text-align: center;
}

.page-header h2 {
    margin: 0 0 8px 0;
    font-size: 28px;
    color: #303133;
}

.page-header p {
    margin: 0;
    color: #909399;
    font-size: 14px;
}

.profile-content {
    display: grid;
    grid-template-columns: 300px 1fr;
    gap: 24px;
}

.profile-left {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.avatar-card {
    text-align: center;
}

.avatar-section {
    margin-bottom: 20px;
}

.user-avatar {
    margin-bottom: 16px;
    border: 3px solid #f0f2f5;
    font-size: 48px !important;
    font-weight: 600;
}

.avatar-actions {
    display: flex;
    gap: 8px;
    justify-content: center;
}

.basic-info h3 {
    margin: 0 0 8px 0;
    font-size: 18px;
    color: #303133;
}

.username {
    margin: 0 0 8px 0;
    color: #606266;
    font-size: 14px;
}

.identity {
    margin: 0 0 12px 0;
}

.join-time {
    margin: 0;
    color: #909399;
    font-size: 12px;
}

.profile-right {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.password-info {
    color: #606266;
    font-size: 14px;
}
.action-buttons {
    display: flex;
}

.last-update {
    margin-top: 8px;
    font-size: 12px;
    color: #909399;
}

.avatar-upload {
    text-align: center;
}

.avatar-uploader {
    display: inline-block;
    border: 2px dashed #d9d9d9;
    border-radius: 6px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
    width: 100px;
    height: 100px;
    line-height: 100px;
}

.avatar-uploader:hover {
    border-color: #409eff;
}

.avatar-uploader-icon {
    font-size: 28px;
    color: #8c939d;
}

.upload-tips {
    margin-top: 16px;
    color: #909399;
    font-size: 12px;
}

.upload-tips p {
    margin: 4px 0;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .profile-content {
        grid-template-columns: 1fr;
    }
    
    .profile-left {
        order: 2;
    }
    
    .profile-right {
        order: 1;
    }
    
    .avatar-actions {
        flex-direction: column;
    }
}
</style>