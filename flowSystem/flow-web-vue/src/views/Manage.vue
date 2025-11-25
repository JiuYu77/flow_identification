<template>
    <div class="admin-layout">
        <!-- 侧边栏 -->
        <div class="sidebar" :class="{ 'sidebar-collapsed': isSidebarCollapsed }">
            <div class="sidebar-header">
                <transition name="fade" mode="out-in">
                    <div v-if="!isSidebarCollapsed" class="sidebar-title">
                        <h2>流型识别系统</h2>
                        <p>管理后台</p>
                    </div>
                    <div v-else class="sidebar-icon">
                        <div class="system-icon">F</div>
                    </div>
                </transition>
                <div class="sidebar-toggle" @click="toggleSidebar">
                    <el-icon v-if="isSidebarCollapsed"><Expand /></el-icon>
                    <el-icon v-else><Fold /></el-icon>
                </div>
            </div>
            
            <el-menu
                class="sidebar-menu"
                :default-active="activeMenu"
                background-color="#304156"
                text-color="#bfcbd9"
                active-text-color="#409eff"
                router
                :collapse="isSidebarCollapsed"
                :collapse-transition="false"
            >
                <el-menu-item index="/dashboard">
                    <el-icon><House /></el-icon>
                    <template #title>
                        <span>仪表盘</span>
                    </template>
                </el-menu-item>
                
                <el-menu-item index="/flow-analysis">
                    <el-icon><DataAnalysis /></el-icon>
                    <template #title>
                        <span>流型分析</span>
                    </template>
                </el-menu-item>
                
                <el-menu-item index="/data-management">
                    <el-icon><Folder /></el-icon>
                    <template #title>
                        <span>数据管理</span>
                    </template>
                </el-menu-item>
                
                <!-- ✅ 新增地址管理菜单项 -->
                <el-menu-item index="/address-management">
                    <el-icon><Location /></el-icon>
                    <template #title>
                        <span>地址管理</span>
                    </template>
                </el-menu-item>
                
                <!-- ✅ 新增用户管理菜单项 - 仅管理员可见 -->
                <el-menu-item v-if="isAdmin" index="/user-management">
                    <el-icon><User /></el-icon>
                    <template #title>
                        <span>用户管理</span>
                    </template>
                </el-menu-item>
                
                <el-menu-item index="/flow-science">
                    <el-icon><Reading /></el-icon>
                    <template #title>
                        <span>流型科普</span>
                    </template>
                </el-menu-item>
            </el-menu>
        </div>

        <!-- 主内容区域 -->
        <div class="main-content" :class="{ 'main-content-expanded': isSidebarCollapsed }">
            <!-- 顶部导航栏 -->
            <div class="top-navbar">
                <div class="navbar-left">
                    <!-- 侧边栏切换按钮（移动端显示） -->
                    <el-button 
                        class="mobile-sidebar-toggle" 
                        @click="toggleSidebar"
                        :icon="isSidebarCollapsed ? Expand : Fold"
                        circle
                        size="small"
                    />
                    <!-- 面包屑导航 -->
                    <el-breadcrumb separator="/">
                        <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
                        <el-breadcrumb-item>{{ currentRouteName }}</el-breadcrumb-item>
                    </el-breadcrumb>
                </div>
                
                <div class="navbar-right">
                    <div class="user-info">
                        <el-dropdown @command="handleUserCommand">
                            <span class="el-dropdown-link">
                                <el-avatar :src="user.avatar">{{ user.nickname?.charAt(0) || user.username?.charAt(0) || 'U' }}</el-avatar>
                                <span class="user-name">{{ user.nickname || user.username || user.email || "未设置用户名" }}</span>
                                <el-icon><ArrowDown /></el-icon>
                            </span>
                            <template #dropdown>
                                <el-dropdown-menu>
                                    <el-dropdown-item command="profile">
                                        <el-icon><User /></el-icon>
                                        个人信息
                                    </el-dropdown-item>
                                    <el-dropdown-item command="settings">
                                        <el-icon><Setting /></el-icon>
                                        设置
                                    </el-dropdown-item>
                                    <el-dropdown-item divided command="logout">
                                        <el-icon><SwitchButton /></el-icon>
                                        退出登录
                                    </el-dropdown-item>
                                </el-dropdown-menu>
                            </template>
                        </el-dropdown>
                    </div>
                </div>
            </div>

            <!-- 内容区域 -->
            <div class="content-wrapper">
                <router-view /><!-- 路由视图 -->
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useUserStore, Admin_Identity } from '@/store/user.js'
import { ElMessageBox } from 'element-plus'

// 图标导入
import {
    House,
    DataAnalysis,
    Folder,
    Setting,
    ArrowDown,
    Expand,
    Fold,
    User,
    SwitchButton,  // ✅ 新增退出登录图标
    Location,  // ✅ 新增地址管理图标
    Reading  // ✅ 新增流型科普图标
} from '@element-plus/icons-vue'

const user = useUserStore()
const route = useRoute()
const router = useRouter()
const isSidebarCollapsed = ref(false)

// ✅ 修复：使用计算属性动态获取当前激活菜单
const activeMenu = computed(() => {
    return route.path
})

// ✅ 判断是否是管理员
const isAdmin = computed(() => {
    return user.identity === Admin_Identity
})

// 响应式设计 - 移动端自动折叠侧边栏
const isMobile = ref(false)

// 检查屏幕尺寸
const checkScreenSize = () => {
    isMobile.value = window.innerWidth < 768
    if (isMobile.value) {
        isSidebarCollapsed.value = true
    }
}

// 切换侧边栏
const toggleSidebar = () => {
    isSidebarCollapsed.value = !isSidebarCollapsed.value
}

// 计算当前路由名称
const currentRouteName = computed(() => {
    const routeMap = {
        '/dashboard': '仪表盘',
        '/flow-analysis': '流型分析',
        '/data-management': '数据管理',
        '/address-management': '地址管理',  // ✅ 新增地址管理路由映射
        '/user-management': '用户管理',  // ✅ 新增用户管理路由映射
        '/flow-science': '流型科普',  // ✅ 修改为流型科普
        '/user-profile': '个人信息',  // ✅ 新增个人信息路由映射
        '/user-settings': '用户设置'  // ✅ 新增用户设置路由映射
    }
    return routeMap[route.path] || '仪表盘'
})

// 用户操作处理
const handleUserCommand = async (command) => {
    switch (command) {
        case 'logout':
            await confirmLogout()
            break
        case 'profile':
            // ✅ 跳转到个人信息页面
            router.push('/user-profile')
            break
        case 'settings':
            // 跳转到用户设置页面
            router.push('/user-settings')
            break
    }
}

// 退出登录确认
const confirmLogout = async () => {
    try {
        await ElMessageBox.confirm(
            '确定要退出登录吗？',
            '退出确认',
            {
                confirmButtonText: '确定',
                cancelButtonText: '取消',
                type: 'warning',
                customClass: 'confirm-dialog',
            }
        )
        handleLogout()
    } catch (error) {
        // 用户取消退出
    }
}

// 退出登录处理
const handleLogout = async () => {
    user.logout()
}

// 组件挂载时检查屏幕尺寸
onMounted(() => {
    checkScreenSize()
    window.addEventListener('resize', checkScreenSize)
})

// 组件卸载时移除事件监听
onUnmounted(() => {
    window.removeEventListener('resize', checkScreenSize)
})
</script>

<style scoped>
/* 管理页面样式已提取到单独文件 */

/* 侧边栏头部动画 */
.fade-enter-active,
.fade-leave-active {
    transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
    opacity: 0;
}
</style>