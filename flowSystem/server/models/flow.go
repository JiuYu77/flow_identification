package model

import (
	"time"

	"gorm.io/gorm"
)

type FlowPattern struct {
	ID        uint           `gorm:"primarykey" json:"id"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	DeletedAt gorm.DeletedAt `gorm:"index" json:"deleted_at"` // GORM框架的标准软删除字段（逻辑删除），软删除时会设置该字段

	PatternType    string    `json:"patternType"` // 流型类型 段塞流。。。
	PredictedLabel int       `json:"predictedLabel"`
	Prob           float64   `json:"prob"`
	Status         int       `json:"status"`                                // 识别状态 0 错误 1 正确
	Data           []float64 `gorm:"type:json;serializer:json" json:"data"` // 原始数据，存储为 JSON 字符串
	Addr           string    `json:"addr"`                                  // 流型分析地址
}

func (f *FlowPattern) TableName() string {
	return "flow_pattern"
}
