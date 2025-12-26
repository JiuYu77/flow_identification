<template>
    <div class="login-container">
        <div class="login-box">
            <div class="login-header">
                <div class="system-logo">
                    <img src="/public/favicon.svg" alt="流型识别系统" class="logo-img" @error="handleImageError">
                </div>
                <h1 class="login-title">欢迎来到流型识别系统</h1>
                <p class="login-subtitle">请登录您的账户</p>
            </div>
            
            <div class="login-form">
                <el-form label-width="0" size="large">
                    <el-form-item>
                        <el-input 
                            v-model="userInfo.username.value" 
                            placeholder="请输入用户名或邮箱"
                            :prefix-icon="User"
                            clearable
                        ></el-input>
                    </el-form-item>

                    <el-form-item>
                        <el-input 
                            v-model="userInfo.password.value" 
                            placeholder="请输入密码" 
                            type="password"
                            :prefix-icon="Lock"
                            show-password
                            clearable
                            @keyup.enter="handleLogin"
                        ></el-input>
                    </el-form-item>
                    <el-form-item>
                        <el-button 
                            type="primary" 
                            size="large" 
                            @click="handleLogin"
                            class="login-button"
                            :loading="loading"
                        >
                            <template #loading>
                                <el-icon class="loading-spin"><Loading /></el-icon>
                                登录中...
                            </template>
                            登录
                        </el-button>
                    </el-form-item>

                    <!-- 忘记密码链接 -->
                    <div class="forgot-password-container">
                        <el-link 
                            type="primary" 
                            class="forgot-password-link"
                            @click="showForgotPasswordDialog"
                            :underline="false"
                        >
                            <el-icon><Key /></el-icon>
                            忘记密码？
                        </el-link>
                    </div>
                </el-form>
            </div>
            
            <div class="login-footer">
                <p class="copyright">© 2025 流型识别系统 版权所有</p>
            </div>
        </div>

        <!-- 忘记密码对话框 -->
        <el-dialog
            v-model="forgotPasswordDialogVisible"
            width="500px"
            :close-on-click-modal="false"
            :close-on-press-escape="false"
            class="forgot-password-dialog"
            align-center
        >
            <template #header>
                <div class="dialog-header-custom">
                    <div class="dialog-icon">
                        <el-icon size="24"><Key /></el-icon>
                    </div>
                    <div class="dialog-title">
                        <h3>重置密码</h3>
                        <p class="dialog-subtitle">请按照以下步骤重置您的密码</p>
                    </div>
                </div>
            </template>
            
            <div class="dialog-content">
                <el-form 
                    :model="forgotPasswordForm" 
                    :rules="forgotPasswordRules" 
                    ref="forgotPasswordFormRef"
                    label-width="0"
                    size="large"
                >
                    <!-- 步骤1：输入用户名或邮箱 -->
                    <div v-if="resetStep === 1" class="reset-step">
                        <div class="step-indicator">
                            <span class="step-number active">1</span>
                            <span class="step-text">验证身份</span>
                        </div>
                        <el-form-item prop="username">
                            <el-input 
                                v-model="forgotPasswordForm.username" 
                                placeholder="请输入用户名或注册邮箱"
                                prefix-icon="User"
                                clearable
                                class="custom-input"
                                autocomplete="off"
                                autocorrect="off"
                                autocapitalize="off"
                                spellcheck="false"
                            >
                                <template #prefix>
                                    <el-icon class="input-icon"><User /></el-icon>
                                </template>
                            </el-input>
                        </el-form-item>
                    </div>
                    
                    <!-- 步骤2：输入验证码和新密码 -->
                    <div v-if="resetStep === 2" class="reset-step">
                        <div class="step-indicator">
                            <span class="step-number active">2</span>
                            <span class="step-text">设置新密码</span>
                        </div>
                        
                        <!-- 验证码发送提示 -->
                        <div class="verification-tip" v-if="verificationSent">
                            <span class="tip-text">我们已向邮箱 {{ maskedEmail }} 发送了验证码，请查收</span>
                        </div>
                        
                        <!-- 验证码输入 -->
                        <el-form-item prop="verification_code">
                            <el-input 
                                v-model="forgotPasswordForm.verification_code" 
                                placeholder="请输入邮箱验证码"
                                prefix-icon="Message"
                                clearable
                                class="custom-input verification-code-input"
                                autocomplete="off"
                                autocorrect="off"
                                autocapitalize="off"
                                spellcheck="false"
                            >
                                <template #prefix>
                                    <el-icon class="input-icon"><Message /></el-icon>
                                </template>
                                <template #suffix>
                                    <el-button 
                                        type="text" 
                                        @click="sendVerificationCode1"
                                        :disabled="countdown > 0"
                                        class="send-code-btn"
                                    >
                                        {{ countdown > 0 ? `${countdown}s后重发` : '发送验证码' }}
                                    </el-button>
                                </template>
                            </el-input>
                        </el-form-item>
                        
                        <!-- 新密码 -->
                        <el-form-item prop="new_password">
                            <el-input 
                                v-model="forgotPasswordForm.new_password" 
                                type="password"
                                placeholder="请输入新密码"
                                prefix-icon="Lock"
                                show-password
                                clearable
                                class="custom-input"
                                autocomplete="new-password"
                                autocorrect="off"
                                autocapitalize="off"
                                spellcheck="false"
                            >
                                <template #prefix>
                                    <el-icon class="input-icon"><Lock /></el-icon>
                                </template>
                            </el-input>
                        </el-form-item>
                        
                        <!-- 确认新密码 -->
                        <el-form-item prop="confirm_password">
                            <el-input 
                                v-model="forgotPasswordForm.confirm_password" 
                                type="password"
                                placeholder="请确认新密码"
                                prefix-icon="Lock"
                                show-password
                                clearable
                                class="custom-input"
                                autocomplete="new-password"
                                autocorrect="off"
                                autocapitalize="off"
                                spellcheck="false"
                            >
                                <template #prefix>
                                    <el-icon class="input-icon"><Lock /></el-icon>
                                </template>
                            </el-input>
                        </el-form-item>
                    </div>
                </el-form>
            </div>
            
            <template #footer>
                <div class="dialog-footer-custom">
                    <el-button 
                        v-if="resetStep === 2"
                        @click="handleBackStep" 
                        size="large"
                        class="back-btn"
                    >
                        <el-icon><ArrowLeft /></el-icon>
                        上一步
                    </el-button>
                    
                    <el-button 
                        @click="cancelForgotPassword" 
                        size="large"
                        class="cancel-btn"
                    >
                        取消
                    </el-button>
                    
                    <el-button 
                        v-if="resetStep === 1"
                        type="primary" 
                        @click="verifyIdentity" 
                        :loading="forgotPasswordLoading"
                        size="large"
                        class="submit-btn"
                    >
                        下一步
                    </el-button>
                    
                    <el-button 
                        v-if="resetStep === 2"
                        type="primary" 
                        @click="submitNewPassword" 
                        :loading="forgotPasswordLoading"
                        size="large"
                        class="submit-btn"
                    >
                        确认重置
                    </el-button>
                </div>
            </template>
        </el-dialog>
    </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue';
import { useUserStore } from '@/store/user.js';
import { storeToRefs } from 'pinia';
import { ElMessage } from 'element-plus';
import { Loading, Key, Message, User, Lock, ArrowLeft } from '@element-plus/icons-vue';
import { resetPassword, sendVerificationCode } from '@/api/user';


const user = useUserStore();
const userInfo = storeToRefs(user);
const loading = userInfo.loading;

// 忘记密码相关状态
const forgotPasswordDialogVisible = ref(false);
const forgotPasswordLoading = ref(false);
const forgotPasswordFormRef = ref();
const resetStep = ref(1); // 重置步骤：1-验证身份，2-设置新密码
const countdown = ref(0); // 验证码倒计时
const verificationSent = ref(false); // 验证码是否已发送

// 忘记密码表单数据
const forgotPasswordForm = reactive({
    username: '', // 用户名或邮箱
    verification_code: '',
    new_password: '',
    confirm_password: '',
    email: ''
});

// 计算属性：获取掩码后的邮箱
const maskedEmail = computed(() => {
    const email = forgotPasswordForm.email;
    if(email.includes('*')){ // 已经掩码过了
        return email;
    }
    if (!email || !email.includes('@')) return '***';
    
    const [username, domain] = email.split('@');
    const maskedUsername = username.length > 2 
        ? username.substring(0, 2) + '*'.repeat(username.length - 2)
        : '*'.repeat(username.length);
    
    return `${maskedUsername}@${domain}`;
});

// 验证密码是否一致
const validateConfirmPassword = (rule, value, callback) => {
    if (value !== forgotPasswordForm.new_password) {
        callback(new Error('两次输入的密码不一致'));
    } else {
        callback();
    }
};

// 忘记密码表单验证规则
const forgotPasswordRules = {
    username: [
        { required: true, message: '请输入用户名或邮箱', trigger: 'blur' }
    ],
    verification_code: [
        { required: true, message: '请输入验证码', trigger: 'blur' },
        { min: 6, max: 6, message: '验证码为6位数字', trigger: 'blur' }
    ],
    new_password: [
        { required: true, message: '请输入新密码', trigger: 'blur' },
        { min: 8, message: '密码长度不能少于8位', trigger: 'blur' }
    ],
    confirm_password: [
        { required: true, message: '请确认新密码', trigger: 'blur' },
        { validator: validateConfirmPassword, trigger: 'blur' }
    ]
};

function handleLogin() {
    if(userInfo.username.value == "" || userInfo.password.value == ""){
        ElMessage.warning({message: "用户名或密码不能为空", plain: true});
        return;
    }

    loading.value = true;
    console.log('handleLogin', useUserStore());
    user.login();
}

// 显示忘记密码对话框
function showForgotPasswordDialog() {
    // 重置状态
    resetStep.value = 1;
    //countdown.value = 0; // 这里不需要重置
    verificationSent.value = false;
    
    // 确保所有输入框为空
    Object.keys(forgotPasswordForm).forEach(key => {
        forgotPasswordForm[key] = '';
    });
    
    // 重置表单验证状态
    if (forgotPasswordFormRef.value) {
        forgotPasswordFormRef.value.clearValidate();
    }
    
    forgotPasswordDialogVisible.value = true;
    
    // 延迟再次清空，防止浏览器自动填充
    setTimeout(() => {
        Object.keys(forgotPasswordForm).forEach(key => {
            forgotPasswordForm[key] = '';
        });
    }, 100);
}

// 取消忘记密码
function cancelForgotPassword() {
    forgotPasswordDialogVisible.value = false;
    resetStep.value = 1;
    //countdown.value = 0; // 这里不需要重置
    verificationSent.value = false;
    
    // 确保所有输入框为空
    Object.keys(forgotPasswordForm).forEach(key => {
        forgotPasswordForm[key] = '';
    });
    
    // 重置表单验证状态
    if (forgotPasswordFormRef.value) {
        forgotPasswordFormRef.value.clearValidate();
    }
}

// 验证身份
async function verifyIdentity() {
    if (!forgotPasswordFormRef.value) return;
    
    try {
        // 验证用户名/邮箱字段
        await forgotPasswordFormRef.value.validateField('username');
        
        forgotPasswordLoading.value = true;
        
        // 验证成功，进入下一步前清空验证码和新密码字段
        forgotPasswordForm.verification_code = '';
        forgotPasswordForm.new_password = '';
        forgotPasswordForm.confirm_password = '';

        // 进入下一步
        resetStep.value = 2;
        verificationSent.value = true;

        // 自动发送验证码
        sendVerificationCode1();
        
    } catch (error) {
        console.error('验证身份失败:', error);
    } finally {
        forgotPasswordLoading.value = false;
    }
}

// 上一步按钮功能 - 添加返回步骤1时的清空逻辑
function handleBackStep() {
    // 清空验证码和新密码字段
    forgotPasswordForm.verification_code = '';
    forgotPasswordForm.new_password = '';
    forgotPasswordForm.confirm_password = '';
    
    // 返回步骤1
    resetStep.value = 1;
    verificationSent.value = false;
    
    // 清除表单验证状态
    if (forgotPasswordFormRef.value) {
        forgotPasswordFormRef.value.clearValidate();
    }
}

// 发送验证码
async function sendVerificationCode1() {
    if (countdown.value > 0) return;
    
    try {
        // 开始倒计时
        countdown.value = 60;
        const timer = setInterval(() => {
            countdown.value--;
            if (countdown.value <= 0) {
                clearInterval(timer);
            }
        }, 1000);
        
        const res = await sendVerificationCode(forgotPasswordForm);
        console.log('sendVerificationCode', res);
        if (res.code !== 0) {
            ElMessage.error(res.msg || '发送验证码失败，请稍后重试');
            return;
        }

        forgotPasswordForm.email = res.data.masked_email;

        ElMessage.success(`验证码已发送至邮箱 ${maskedEmail.value}，请查收`);
    } catch (error) {
        console.error('发送验证码失败:', error);
        ElMessage.error('发送验证码失败，请稍后重试');
    }
}

// 提交新密码
async function submitNewPassword() {
    if (!forgotPasswordFormRef.value) return;
    
    try {
        // 验证所有字段
        const valid = await forgotPasswordFormRef.value.validate();
        if (!valid) return;
        
        forgotPasswordLoading.value = true;
        
        const res = await resetPassword(forgotPasswordForm);
        if (res.code !== 0) {
            ElMessage.error(res.msg || '重置密码失败，请稍后重试');
            return;
        }
        
        // 显示成功消息
        ElMessage.success({
            message: '密码重置成功，请使用新密码登录',
            duration: 5000
        });
        
        // 关闭对话框并重置所有状态
        forgotPasswordDialogVisible.value = false;
        resetStep.value = 1;
        countdown.value = 0;
        verificationSent.value = false;
        Object.keys(forgotPasswordForm).forEach(key => {
            forgotPasswordForm[key] = '';
        });
        
    } catch (error) {
        console.error('重置密码失败:', error);
        ElMessage.error('重置密码失败，请稍后重试');
    } finally {
        forgotPasswordLoading.value = false;
    }
}

// 处理图片加载错误
function handleImageError(event) {
    // 如果 SVG 加载失败，尝试加载 ICO 格式
    const img = event.target;
    if (img.src && !img.src.includes('favicon.ico')) {
        img.src = '/public/favicon.ico';
    } else {
        // 如果都失败，隐藏图片，显示备用文字 "F"
        img.style.display = 'none';
        const logoContainer = img.parentElement;
        if (logoContainer && !logoContainer.querySelector('.fallback-text')) {
            const fallback = document.createElement('span');
            fallback.className = 'fallback-text';
            fallback.textContent = 'F';
            logoContainer.appendChild(fallback);
        }
    }
}
</script>