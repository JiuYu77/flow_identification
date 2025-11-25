import { createApp } from 'vue'
import App from './App.vue'
// import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import router from './router';
import '@/assets/sass/main.scss'

import { createPinia } from 'pinia';

const app = createApp(App)

// app.use(ElementPlus)
app.use(router)
const pinia = createPinia();
app.use(pinia);

app.mount('#app')

