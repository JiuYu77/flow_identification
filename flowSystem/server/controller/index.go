package controller

import (
	"github.com/gin-gonic/gin"
)

func NoRoute(ctx *gin.Context) {
	ctx.File("dist/index.html")
}
