package repo

import (
	model "flow-server/models"
)

func FetchFlowPatternList(keyword string, status int, page int, pageSize int) ([]*model.FlowPattern, int64, error) {
	var flowPatterns []*model.FlowPattern
	var total int64

	// 构建查询条件
	query := DB.Model(&model.FlowPattern{})
	if keyword != "" {
		query = query.Where("pattern_type LIKE ? OR addr LIKE ?", "%"+keyword+"%", "%"+keyword+"%")
	}
	if status != -1 {
		query = query.Where("status = ?", status)
	}

	// 获取总数
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	// 分页查询
	query.Offset((page - 1) * pageSize).Limit(pageSize).Find(&flowPatterns)
	if err := query.Error; err != nil {
		return nil, 0, err
	}
	return flowPatterns, total, nil
}

func GetFlowPatternByID(id uint) (*model.FlowPattern, error) {
	var flowPattern model.FlowPattern
	if err := DB.Where("id = ?", id).First(&flowPattern).Error; err != nil {
		return nil, err
	}
	return &flowPattern, nil
}

// 不会忽略零值字段（如空字符串、0、false），会更新所有字段。
func UpdateFlowPattern(flowPattern *model.FlowPattern) error {
	if err := DB.Save(flowPattern).Error; err != nil {
		return err
	}
	return nil
}

func DeleteFlowPattern(flowPattern *model.FlowPattern) error {
	return DB.Delete(flowPattern).Error // 逻辑删除
}

func BatchDeleteFlowPattern(flowIDs []uint) error {
	return DB.Unscoped().Delete(&model.FlowPattern{}, "id IN ?", flowIDs).Error // 永久删除
}

// CreateFlowPattern 创建流型分析
func CreateFlowPattern(flowPattern *model.FlowPattern) error {
	if err := DB.Create(flowPattern).Error; err != nil {
		return err
	}
	return nil
}
