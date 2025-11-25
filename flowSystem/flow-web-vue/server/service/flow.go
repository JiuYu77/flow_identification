package service

import (
	model "flow-server/models"
	repo "flow-server/repository"
	"flow-server/resp"
	"fmt"

	"github.com/gin-gonic/gin"
)

type FlowListRequest struct {
	Keyword  string `json:"keyword"`
	Status   int    `json:"status"`
	Page     int    `json:"page"`
	PageSize int    `json:"page_size"`
}

func GetFlowPatternList(c *gin.Context) {
	var req FlowListRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "获取流型分析列表失败,参数错误")
		return
	}

	flowList, total, err := repo.FetchFlowPatternList(req.Keyword, req.Status, req.Page, req.PageSize)
	if err != nil {
		resp.FailWithMsg(c, "获取流型分析列表失败")
		return
	}
	resp.OKWithData(c, gin.H{
		"flow_pattern_list": flowList,
		"total":             total,
	})
}

type UpdateFlowPatternRequest struct {
	ID             uint   `json:"id"`
	PatternType    string `json:"pattern_type"`
	PredictedLabel int    `json:"predicted_label"`
	Status         int    `json:"status"`
}

func UpdateFlowPattern(c *gin.Context) {
	var req UpdateFlowPatternRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "更新流型分析失败,参数错误")
		return
	}

	flowPattern, err := repo.GetFlowPatternByID(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "更新流型分析失败,流型分析不存在")
		return
	}

	if req.PatternType != "" {
		flowPattern.PatternType = req.PatternType
	}
	flowPattern.PredictedLabel = req.PredictedLabel
	flowPattern.Status = req.Status

	if err := repo.UpdateFlowPattern(flowPattern); err != nil {
		resp.FailWithMsg(c, "更新流型分析失败")
		return
	}
	resp.OKWithMsg(c, "更新流型分析成功")
}

type DeleteFlowPatternRequest struct {
	ID uint `json:"id"`
}

func DeleteFlowPattern(c *gin.Context) {
	var req DeleteFlowPatternRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "删除流型分析失败,参数错误")
		return
	}

	flowPattern, err := repo.GetFlowPatternByID(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "删除流型分析失败,流型分析不存在")
		return
	}

	if err := repo.DeleteFlowPattern(flowPattern); err != nil {
		resp.FailWithMsg(c, "删除流型分析失败")
		return
	}
	resp.OKWithMsg(c, "删除流型分析成功")
}

type BatchDeleteFlowPatternRequest struct {
	FlowIDs []uint `json:"flow_ids"`
}

func BatchDeleteFlowPattern(c *gin.Context) {
	var req BatchDeleteFlowPatternRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "批量删除流型分析失败,参数错误")
		return
	}

	if err := repo.BatchDeleteFlowPattern(req.FlowIDs); err != nil {
		resp.FailWithMsg(c, "批量删除流型分析失败")
		return
	}
	resp.OKWithMsg(c, "批量删除流型分析成功")
}

type CreateFlowPatternRequest struct {
	PatternType    string    `json:"patternType"` // 流型类型 段塞流。。。
	PredictedLabel int       `json:"predictedLabel"`
	Prob           float64   `json:"prob"`
	Data           []float64 `json:"data"` // 原始数据，存储为 JSON 字符串
	Addr           string    `json:"addr"` // 流型分析地址
}

func CreateFlowPattern(c *gin.Context) {
	var req CreateFlowPatternRequest
	err := c.ShouldBindJSON(&req)
	fmt.Println(">>>>>>>.......", err)
	if err != nil {
		resp.FailWithMsg(c, "创建流型分析失败,参数错误")
		return
	}

	flowPattern := &model.FlowPattern{
		PatternType:    req.PatternType,
		PredictedLabel: req.PredictedLabel,
		Prob:           req.Prob,
		Status:         1, // 初始状态为正确
		Data:           req.Data,
		Addr:           req.Addr,
	}

	if err := repo.CreateFlowPattern(flowPattern); err != nil {
		resp.FailWithMsg(c, "流型数据上传失败")
		return
	}
	resp.OKWithMsg(c, "流型数据上传成功")
}
