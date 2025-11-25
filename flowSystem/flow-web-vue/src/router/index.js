import { createRouter, createWebHistory } from 'vue-router'
import { useUserStore, Admin_Identity } from '@/store/user.js'
import authStorage from '@/store/authStorage'

const router = createRouter({
    history: createWebHistory(),
    routes:[
        {
            path:'/',
            component:()=>import('@/views/Manage.vue'),
            meta: { requiresAuth: true }, // 需要登录才能访问
            redirect: '/dashboard', // 默认重定向到仪表盘
            children: [
                {
                    path:'/dashboard',
                    component:()=>import('@/components/manage/Dashboard.vue')
                },
                {
                    path:'/flow-analysis',
                    component:()=>import('@/components/manage/FlowAnalysis.vue')
                },
                {
                    path:'/data-management',
                    component:()=>import('@/components/manage/DataManagement.vue')
                },
                {
                    path:'/flow-science',
                    component:()=>import('@/components/manage/FlowScience.vue')
                },
                {
                    path:'/user-management',
                    component:()=>import('@/components/manage/UserManagement.vue')
                },
                {
                    path:'/user-profile',
                    component:()=>import('@/views/UserProfile.vue')
                },
                {
                    path:'/user-settings',
                    component:()=>import('@/views/UserSettings.vue')
                },
                {
                    path: '/address-management',
                    name: 'AddressManagement',
                    component: () => import('@/components/manage/AddressManagement.vue'),
                    meta: { title: '地址管理', requiresAuth: true }
                }
            ]
        },
        {
            path:'/login',
            component:()=>import('@/views/Login.vue') // 登录页
        },
    ]
})

let loginCount = 0; // 防止重复进入 /login
// 路由守卫
router.beforeEach(async (to, from, next) => {
    const userStore = useUserStore()

    // 保存刷新前的路径
    if (from.path !== '/login' && to.path !== '/login') {
        authStorage.setPreAuthPath(to.path)
    }

    if(to.meta.requiresAuth && (!userStore.isAuthenticated || authStorage.isAccessTokenExpired())) {
        await userStore.initializeAuth()
    }

    if(to.path == '/login'){
        loginCount++
        if(userStore.isAuthenticated){
            next(authStorage.getPreAuthPath() || '/')
            return
        }else if(loginCount < 2){
            await userStore.initializeAuth()
        }
    }

    if (to.meta.requiresAuth && !userStore.isAuthenticated) {
        loginCount++
        // 如果需要登录但用户未登录，重定向到登录页
        next('/login')
    } else {
        if(to.path == '/user-management' && userStore.getIdentity != Admin_Identity){
            return;  // 非管理员 不能进入
        }
        next()
    }
})

export default router;