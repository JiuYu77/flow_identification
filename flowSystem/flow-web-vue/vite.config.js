import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// 按需导入 Element Plus 组件
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'


// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      resolvers: [ElementPlusResolver()],
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  build:{
    rollupOptions:{
      external: ['three', 'three/addons/controls/OrbitControls.js', 'three/webgpu', 'three/tsl'],
      output:{
        manualChunks(id){
          if(id.includes('node_modules')){
            if (id.includes('vue')) {
              if (id.includes('/vue@') || id.includes('/vue/dist/')) {
                return 'vue-core'
              }
              // 将 Vue Router 单独拆分
              if (id.includes('vue-router')) {
                return 'vue-router'
              }
              // 将状态管理库单独拆分（Pinia）
              if (id.includes('pinia')) {
                return 'vue-state'
              }
              // 将 UI 组件库单独拆分
              if (id.includes('element-plus')) {
                return 'ui-library'
              }
              return 'vue-vendor'; // 其他第三方库
            }
          }
        }
      }
    }
  }
})
