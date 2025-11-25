<template>
    <div class="user-settings-container">
        <!-- 页面标题 -->
        <div class="page-header">
            <h2>用户设置</h2>
            <p>管理您的账户设置和偏好</p>
        </div>

        <!-- 主要内容区域 -->
        <div class="settings-content">
            <el-card class="settings-card">
                <template #header>
                    <div class="card-header">
                        <span>账户设置</span>
                    </div>
                </template>

                <!-- 账户信息展示 -->
                <div class="account-info-section">
                    <div class="info-item">
                        <div class="info-label">用户名</div>
                        <div class="info-value">{{ userStore.username || '未设置' }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">邮箱</div>
                        <div class="info-value">{{ userStore.email || '未设置' }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">身份</div>
                        <div class="info-value">
                            <el-tag :type="userStore.identity === 1 ? 'success' : 'info'">
                                {{ userStore.identity === 1 ? '管理员' : '普通用户' }}
                            </el-tag>
                        </div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">注册时间</div>
                        <div class="info-value">{{ formatTime(userStore.created_at) }}</div>
                    </div>
                </div>
            </el-card>

            <!-- 危险操作区域 -->
            <el-card class="settings-card danger-card">
                <template #header>
                    <div class="card-header danger-header">
                        <el-icon><Warning /></el-icon>
                        <span>危险操作</span>
                    </div>
                </template>

                <div class="danger-section">
                    <div class="danger-item">
                        <div class="danger-info">
                            <h3>注销账号</h3>
                            <p>注销后，您的账户将被永久删除，所有数据将无法恢复。请谨慎操作。</p>
                        </div>
                        <el-button 
                            type="danger" 
                            :icon="Delete"
                            @click="handleDeleteAccount"
                            :loading="deleting"
                        >
                            注销账号
                        </el-button>
                    </div>
                </div>
            </el-card>
        </div>
    </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useUserStore } from '@/store/user'
import { deleteUser, getUserProfile } from '@/api/user'
import { Delete, Warning } from '@element-plus/icons-vue'
import { formatTime } from '@/assets/js/time';


const router = useRouter()
const userStore = useUserStore()
const deleting = ref(false)


// 处理注销账号
const handleDeleteAccount = async () => {
    try {
        // 第一次确认：警告对话框
        await ElMessageBox.confirm(
            '注销账号后，您的账户将被永久删除，所有数据将无法恢复。此操作不可撤销！',
            '确认注销账号',
            {
                confirmButtonText: '我已知晓，继续注销',
                cancelButtonText: '取消',
                type: 'error',
                customClass: 'delete-account-dialog',
                dangerouslyUseHTMLString: false
            }
        )

        // 第二次确认：输入用户名确认
        const { value: confirmValue } = await ElMessageBox.prompt(
            `请输入您的用户名 "${userStore.username}" 以确认注销账号`,
            '最后确认',
            {
                confirmButtonText: '确认注销',
                cancelButtonText: '取消',
                inputPlaceholder: '请输入用户名',
                inputValidator: (value) => {
                    if (!value) {
                        return '请输入用户名'
                    }
                    if (value !== userStore.username) {
                        return '用户名不匹配，请重新输入'
                    }
                    return true
                },
                inputType: 'text',
                type: 'warning'
            }
        )

        // 开始注销流程
        deleting.value = true
        
        try {
            const res = await deleteUser({
                id: userStore.id,
                username: userStore.username
            });

            if (res.code === 0) {
                ElMessage.success({
                    message: '账号已成功注销',
                    duration: 2000
                })
                
                // 清除用户数据并跳转到登录页
                userStore.reset()
                router.push('/login')
            } else {
                ElMessage.error({
                    message: res.msg || '注销账号失败',
                    duration: 3000
                })
            }
        } catch (error) {
            console.error('注销账号失败:', error)
            ElMessage.error({
                message: error.message || '注销账号失败，请稍后重试',
                duration: 3000
            })
        } finally {
            deleting.value = false
        }
    } catch (error) {
        // 用户取消操作
        if (error !== 'cancel') {
            console.error('注销账号确认失败:', error)
        }
    }
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
onMounted(() => {
    fetchUserProfile()
})
</script>

<style scoped lang="scss">
/* 更清爽的设置页面样式，统一卡片、间距与危险操作视觉 */
.user-settings-container {
    padding: 0;
    max-width: 1100px;
    margin: 20px auto;
    background: #f6f8fb;
    border-radius: 12px;
}

.page-header {
    margin-bottom: 20px;
    h2 {
        font-size: 26px;
        font-weight: 600;
        color: #111827;
        margin: 0 0 6px 0;
    }
    p {
        font-size: 14px;
        color: #6b7280;
        margin: 0;
    }
}

.settings-content {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.settings-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 8px 20px rgba(16,24,40,0.05);

    .card-header {
        display: flex;
        align-items: center;
        font-size: 16px;
        font-weight: 600;
        color: #111827;

        .el-icon { margin-right: 8px; }

        &.danger-header { color: #e11d48; .el-icon { color: #e11d48; } }
    }
}

.danger-card {
    border: 1px solid #fde2e2;
    background-color: #fff7f7;
    :deep(.el-card__header) { background-color: transparent; }
}

.account-info-section {
    .info-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 0;
        border-bottom: 1px solid #eef2f7;
        &:last-child { border-bottom: none; }

        .info-label { font-size: 14px; color: #4b5563; font-weight: 500; }
        .info-value { font-size: 14px; color: #0f1724; }
    }
}

.danger-section {
    .danger-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px;
        background-color: #fff;
        border-radius: 8px;
        border: 1px solid #fde2e2;

        .danger-info { flex: 1; margin-right: 18px;
            h3 { font-size: 16px; font-weight: 600; color: #111827; margin: 0 0 6px 0; }
            p { font-size: 14px; color: #6b7280; margin: 0; line-height: 1.6; }
        }
    }
}

@media (max-width: 768px) {
    .user-settings-container { padding: 16px; }
    .danger-section .danger-item { flex-direction: column; align-items: flex-start; .danger-info { margin-right: 0; margin-bottom: 12px; } .el-button { width: 100%; } }
}
</style>

