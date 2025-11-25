<template>
    <div class="flowSci-container">
        <div class="flowSci-page-header">
            <h2>流型科普</h2>
        </div>

        <el-card class="flowSci-card">
            <template #header>
                <h3>什么是流型？</h3>
            </template>
            <div class="flowSci-content-layout">
                <div class="flowSci-text-content">
                    <p class="flowSci-intro-text">
                        流型（Flow Pattern）是指在多相流系统中，不同相态（如气、液、固）在管道中流动时所呈现的特定形态和分布模式。
                        根据流体的相态、流速、压力、粘度、密度等因素变化，流体会表现出不同流动特征，形成不同的流型。
                        它是流体力学研究中的重要概念，广泛应用于石油、化工、能源等领域。
                    </p>
                    <div class="flowSci-feature-list">
                        <h4 style="margin: 0;">流型的特性</h4>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>描述流体相态的空间分布</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>影响传热传质效率</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>决定流动阻力和压降</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>影响设备运行稳定性</span>
                        </div>
                    </div>
                    <p class="flowSci-intro-text">
                        下面是流型的一些常见类型。
                    </p>
                </div>
                <div class="flowSci-image-content">
                    <div class="flowSci-image-container">
                        <div class="flowSci-image-group">
                            <div class="flowSci-image-item">
                                <div class="flowSci-image-wrapper">
                                    <img src="/img/experiment.png" alt="实验流型示意图" class="flowSci-image" />
                                </div>
                                <div class="flowSci-image-caption">实验流型示意图</div>
                            </div>
                            <div class="flowSci-image-item">
                                <div class="flowSci-image-wrapper">
                                    <img src="/img/actual-pipe.png" alt="实际管道示意图" class="flowSci-image" />
                                </div>
                                <div class="flowSci-image-caption">实际管道示意图</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </el-card>

        <el-card class="flowSci-card">
            <template #header>
                <h3>段塞流</h3>
            </template>
            <div class="flowSci-content-layout">
                <div class="flowSci-text-content">
                    <p class="flowSci-intro-text">
                        段塞流（Slug Flow）是管道气液两相流动的重要流型之一，其特征表现为气柱与液塞交替出现的间歇性流动结构。
                        在水平管道中，该流型属于间歇流类别，与分离流类别的波状流、环状流共同构成流型谱系。其形成机制与气相流量增大导致小气泡聚并为大气泡密切相关，垂直管道中气柱头部呈现弹状结构。
                    </p>

                    <div class="flowSci-feature-list">
                        <h4 style="margin: 0;">流型的特征</h4>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>导致管道持液率与压力剧烈波动，对海洋立管系统产生周期性应力冲击</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>导致管壁腐蚀、缩短设备的使用寿命</span>
                        </div>
                    </div>
                </div>
            </div>
        </el-card>
        <el-card class="flowSci-card">
            <template #header>
                <h3>伪段塞流</h3>
            </template>
             <div class="flowSci-content-layout">
                <div class="flowSci-text-content">
                    <p class="flowSci-intro-text">
                        伪段塞流（Pseudo Slug Flow）介于分层流和段塞流之间，具有部分段塞流的特征，但又不完全发展。
                        液波生长、破碎，但形成的液塞不稳定，无法持续堵塞整个管道，或者液塞头部不能到达管顶。
                    </p>
                    <div class="flowSci-feature-list">
                        <h4 style="margin: 0;">流型的特征</h4>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>气液界面处产生大振幅的波浪</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>这些波浪会生长、破碎，有时会形成短暂的、未发展完全的液桥，但很快又会崩塌</span>
                        </div>
                    </div>
                </div>
            </div>
        </el-card>
        <el-card class="flowSci-card">
            <template #header>
                <h3>分层光滑（平滑）流</h3>
            </template>
             <div class="flowSci-content-layout">
                <div class="flowSci-text-content">
                    <p class="flowSci-intro-text">
                        分层光滑（平滑）流（Stratified Smooth Flow）气液两相完全分离，界面平整光滑。
                    </p>
                    <div class="flowSci-feature-list">
                        <h4 style="margin: 0;">流型的特征</h4>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>流动平稳，几乎没有波动</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>通常发生在气液两相流速都非常低的情况下，剪切力不足以扰动界面形成波浪</span>
                        </div>
                    </div>
                </div>
            </div>
        </el-card>
        <el-card class="flowSci-card">
            <template #header>
                <h3>分层波浪流</h3>
            </template>
             <div class="flowSci-content-layout">
                <div class="flowSci-text-content">
                    <p class="flowSci-intro-text">
                        分层波浪流（Stratified Wavy Flow）气相和液相在重力作用下完全分离，气体在上，液体在下。
                        气液界面上存在连续的、运动的波浪。
                    </p>
                    <div class="flowSci-feature-list">
                        <h4 style="margin: 0;">流型的特征</h4>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>相比段塞流，流动要稳定得多，但仍有波动</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>清晰的气液分界面</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>界面不再是光滑的，而是布满了不同波长和波高的波浪。这些波浪是由于气液两相相对速度较高时，气流对液面的剪切作用造成的</span>
                        </div>
                    </div>
                </div>
            </div>
        </el-card>
        <el-card class="flowSci-card">
            <template #header>
                <h3>泡沫段塞流</h3>
            </template>
             <div class="flowSci-content-layout">
                <div class="flowSci-text-content">
                    <p class="flowSci-intro-text">
                        泡沫段塞流（Foamy Slug Flow）在这里是指带有泡沫的段塞流。泡沫是某些物质溶于液体形成的。
                        常见于含表面活性剂的体系：例如在油气开采中，某些化学物质（如缓蚀剂）的存在容易导致泡沫生成。
                    </p>
                    <div class="flowSci-feature-list">
                        <h4 style="margin: 0;">流型的特征</h4>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>形态与普通段塞流类似，也是液塞和气弹交替</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>关键区别在于：液塞单元不再是清澈的液体，而是充满了微小气泡的泡沫混合物，外观呈白色或乳白色。</span>
                        </div>
                    </div>
                </div>
            </div>
        </el-card>
        <el-card class="flowSci-card">
            <template #header>
                <h3>分层泡沫波浪流</h3>
            </template>
             <div class="flowSci-content-layout">
                <div class="flowSci-text-content">
                    <p class="flowSci-intro-text">
                        分层泡沫波浪流（Stratified Foamy Wavy Flow）在分层流的基础上，液相中混入了泡沫。
                        仍然是气体在上，液体在下。
                    </p>
                    <div class="flowSci-feature-list">
                        <h4 style="margin: 0;">流型的特征</h4>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>下部的液体是泡沫状的</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>管壁上也存在泡沫</span>
                        </div>
                    </div>
                </div>
            </div>
        </el-card>
        <el-card class="flowSci-card">
            <template #header>
                <h3>泡沫环状流</h3>
            </template>
             <div class="flowSci-content-layout">
                <div class="flowSci-text-content">
                    <p class="flowSci-intro-text">
                        泡沫环状流（Foamy Annular Flow）液体在管壁形成一层液膜，气体在管道中心高速流动。
                        管壁上的液膜中充满了泡沫。
                    </p>
                    <div class="flowSci-feature-list">
                        <h4 style="margin: 0;">流型的特征</h4>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>由于泡沫的存在，液膜体积膨胀，管道中的持液率（液体所占比例）比普通环状流高</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>管道中心是高速流动的气体核心，其中夹带着液滴</span>
                        </div>
                        <div class="flowSci-feature-item">
                            <el-icon><Check /></el-icon>
                            <span>管壁四周是一层厚实的、充满气泡的泡沫状液膜在流动</span>
                        </div>
                    </div>
                </div>
            </div>
        </el-card>
    </div>
</template>

<script setup>
// 图标导入
import { Check } from '@element-plus/icons-vue'
// 流型科普页面逻辑
</script>

<style scoped>
</style>