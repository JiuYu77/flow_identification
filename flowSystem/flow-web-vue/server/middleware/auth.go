package middleware

import (
	"strings"

	"flow-server/jwt"
	"flow-server/resp"

	"github.com/gin-gonic/gin"
)

// AuthMiddleware 验证请求中的 access token，验证通过后在上下文中设置 userID 和 userEmail
func AuthMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// 从 Authorization 头获取 token，支持 "Bearer <token>"
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			resp.FailWithCode(c, 1002)
			c.Abort()
			return
		}

		parts := strings.SplitN(authHeader, " ", 2)
		var tokenString string
		if len(parts) == 2 && strings.ToLower(parts[0]) == "bearer" { // 支持 "Bearer <token>" 格式
			tokenString = parts[1]
		} else {
			tokenString = authHeader
		}

		// 验证 token
		claims, err := jwt.CheckToken(tokenString, "access")
		if err != nil {
			resp.FailWithMsg(c, err.Error())
			c.Abort()
			return
		}

		// 通过，将用户信息存入上下文，供后续 handler 使用
		c.Set("id", claims.ID)       // userID
		c.Set("email", claims.Email) // userEmail
		c.Next()
	}
}
