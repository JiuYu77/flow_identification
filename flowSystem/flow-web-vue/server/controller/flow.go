package controller

import (
	"flow-server/service"

	"github.com/gin-gonic/gin"
)

func GetFlowPatternList(c *gin.Context) {
	service.GetFlowPatternList(c)
}

func UpdateFlowPattern(c *gin.Context) {
	service.UpdateFlowPattern(c)
}

func DeleteFlowPattern(c *gin.Context) {
	service.DeleteFlowPattern(c)
}

func BatchDeleteFlowPattern(c *gin.Context) {
	service.BatchDeleteFlowPattern(c)
}

func CreateFlowPattern(c *gin.Context) {
	service.CreateFlowPattern(c)
}
