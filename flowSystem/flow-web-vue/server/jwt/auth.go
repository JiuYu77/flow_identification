package jwt

import (
	"context"
	"errors"
	repo "flow-server/repository"
	"fmt"
	"strconv"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

var (
	jwtKey             []byte // 定义一个用于签名的 JWT密钥，实际生产中应使用复杂密钥并从安全配置中读取。
	AccessTokenExpire  time.Duration
	RefreshTokenExpire time.Duration
)

// 定义自定义 声明结构体
type Claims struct {
	ID                   uint   `json:"id"`         // 用户ID
	Email                string `json:"email"`      // 用户邮箱
	TokenType            string `json:"token_type"` // 令牌类型 如 "access" 或 "refresh"
	jwt.RegisteredClaims        // 自动检测是否过期，ExpiresAt exp 字段为过期时间
}

// 定义令牌响应结构体
type TokenResponse struct {
	AccessToken  string `json:"access_token"`  // 访问令牌
	RefreshToken string `json:"refresh_token"` // 刷新令牌
	ExpiresIn    int64  `json:"expires_in"`    // 访问令牌 过期时间间隔（秒）
}

// GenerateTokenPair 生成包含 访问令牌 和 刷新令牌 的令牌对
func GenerateTokenPair(id uint, email string) (*TokenResponse, error) {
	// 创建访问令牌声明
	accessClaims := Claims{
		ID:        id,
		Email:     email,
		TokenType: "access",
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(AccessTokenExpire)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
		},
	}

	accessToken := jwt.NewWithClaims(jwt.SigningMethodHS256, accessClaims) // 创建访问令牌
	accessTokenString, err := accessToken.SignedString(jwtKey)
	if err != nil {
		return nil, err
	}

	// 创建刷新令牌声明
	refreshClaims := Claims{
		ID:        id,
		Email:     email,
		TokenType: "refresh",
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(RefreshTokenExpire)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
		},
	}

	refreshToken := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims) // 创建刷新令牌
	refreshTokenString, err := refreshToken.SignedString(jwtKey)
	if err != nil {
		return nil, err
	}

	// 将 token 存储到 Redis
	if err := StoreTokenInRedis(id, accessTokenString, refreshTokenString); err != nil {
		return nil, err
	}

	return &TokenResponse{
		AccessToken:  accessTokenString,
		RefreshToken: refreshTokenString,
		ExpiresIn:    int64(AccessTokenExpire.Seconds()),
	}, nil
}

// 实现将 token 存储到 Redis 的逻辑
func StoreTokenInRedis(id uint, accessToken string, refreshToken string) error {
	ctx := context.Background()
	userKey := fmt.Sprintf("user_tokens:%d", id)

	// 使用 hash 存储用户的token信息，access token 和 refresh token
	err := repo.RedisClient.HSet(
		ctx, userKey,
		"access_token", accessToken, // key1, value1
		"refresh_token", refreshToken,
		"issued_at", time.Now().Unix(), // token的签发时间
	).Err()
	if err != nil {
		return err
	}

	// 设置过期时间为 刷新token 的过期时间
	return repo.RedisClient.Expire(ctx, userKey, RefreshTokenExpire).Err()
}

// 从 Redis 中删除用户的 token 信息
func DeleteTokenFromRedis(id uint) error {
	ctx := context.Background()
	userKey := fmt.Sprintf("user_tokens:%d", id)

	// 删除用户的 token 信息
	return repo.RedisClient.Del(ctx, userKey).Err()
}

// GetTokenIssuedTime 获取用户 token 被签发的时间
func GetTokenIssuedTime(id uint) (time.Time, error) {
	ctx := context.Background()
	userKey := fmt.Sprintf("user_tokens:%d", id)
	issuedAtStr, err := repo.RedisClient.HGet(ctx, userKey, "issued_at").Result()
	if err != nil {
		return time.Time{}, err
	}

	issuedAt, err := strconv.ParseInt(issuedAtStr, 10, 64)
	if err != nil {
		return time.Time{}, err
	}
	return time.Unix(issuedAt, 0), nil
}

// 验证 token 是否有效
// 验证 token 是否与 Redis 中存储的 token 匹配
// true 表示 token 有效
func IsTokenValid(id uint, token, tokenType string) (bool, error) {
	ctx := context.Background()
	userKey := fmt.Sprintf("user_tokens:%d", id)

	// 从 Redis 中获取存储的 token
	var tokenField string
	switch tokenType {
	case "access":
		tokenField = "access_token"
	case "refresh":
		tokenField = "refresh_token"
	default:
		return false, fmt.Errorf("invalid token type")
	}

	redisToken, err := repo.RedisClient.HGet(ctx, userKey, tokenField).Result()
	if err != nil {
		return false, err
	}

	// 比较 token 是否匹配
	return redisToken == token, nil
}

// ParseToken 解析并验证 JWT token 字符串，返回 Claims
func ParseToken(tokenString string) (*Claims, error) {
	var claims Claims
	token, err := jwt.ParseWithClaims(tokenString, &claims, func(t *jwt.Token) (any, error) { // 会检测 token 是否过期
		return jwtKey, nil
	})
	if err != nil {
		return nil, err
	}
	if !token.Valid {
		return nil, fmt.Errorf("invalid token")
	}
	return &claims, nil
}

func CheckToken(token string, tokenType string) (*Claims, error) {
	// 解析并验证 token
	claims, err := ParseToken(token)
	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			if tokenType == "refresh" {
				return nil, errors.New("令牌已过期，请重新登录")
			}
			return nil, errors.New("令牌已过期")
		}
		return nil, errors.New("无效的令牌")
	}

	// 必须是 tokenType
	if claims.TokenType != tokenType {
		return nil, errors.New("令牌类型错误")
	}

	// 验证 token 是否与 Redis 中存储的 token 匹配
	isValid, err := IsTokenValid(claims.ID, token, tokenType)
	if err != nil || !isValid {
		return nil, errors.New("令牌无效")
	}
	return claims, nil
}
