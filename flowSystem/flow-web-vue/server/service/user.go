package service

import (
	"flow-server/jwt"
	model "flow-server/models"
	repo "flow-server/repository"
	"flow-server/resp"
	"flow-server/utils"
	"fmt"
	"mime/multipart"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
)

type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type LoginResponse struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	ExpiresIn    int64     `json:"expires_in"`
	User         *UserInfo `json:"user"`
}

type UserInfo struct {
	ID        uint   `json:"id"`
	Email     string `json:"email"`
	Username  string `json:"username"`
	Nickname  string `json:"nickname"`
	Avatar    string `json:"avatar"`
	Identity  int    `json:"identity"` // 身份，如 2 普通用户、1 管理员等
	CreatedAt string `json:"created_at"`
	UpdatedAt string `json:"updated_at"`
}

func Login(c *gin.Context) {
	var req LoginRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "登录失败")
		return
	}

	// 从数据库中查询用户
	user, err := repo.GetUserByEmail(req.Username)
	if err != nil {
		user, err = repo.GetUserByUsername(req.Username)
		if err != nil {
			resp.FailWithMsg(c, "登录失败，用户名或密码错误")
			return
		}
	}

	// 验证密码
	if !CheckPassword(req.Password, user.Password) {
		resp.FailWithMsg(c, "登录失败，用户名或密码错误")
		return
	}

	// 生成 jwt token
	token, err := jwt.GenerateTokenPair(user.ID, user.Email)
	if err != nil {
		resp.FailWithMsg(c, "登录失败")
		return
	}

	loginResp := LoginResponse{
		AccessToken:  token.AccessToken,
		RefreshToken: token.RefreshToken,
		ExpiresIn:    token.ExpiresIn,
		User: &UserInfo{
			ID:       user.ID,
			Email:    user.Email,
			Nickname: user.Nickname,
			Avatar:   user.Avatar,
			Identity: user.Identity,
		},
	}

	setRefreshTokenCookie(c, token.RefreshToken)

	// 验证通过，返回用户信息
	resp.OKWithData(c, loginResp)
}

type AuthRequest struct {
	ID            uint   `json:"id"`
	Authorization string `json:"Authorization"` // access_token
	TokenType     string `json:"token_type"`
}
type AuthResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
}

func Validate(c *gin.Context) {
	_, err := CheckAccessToken(c)
	if err != nil {
		resp.FailWithMsg(c, err.Error())
		return
	}

	resp.OKWithMsg(c, "令牌有效")
}

// 刷新令牌
func Refresh(c *gin.Context) {
	var req AuthRequest
	var err error
	req.ID, err = getUserIDFromHeader(c)
	if err != nil {
		resp.FailWithMsg(c, "获取用户ID失败")
		return
	}

	claims, err := CheckRefreshToken(c)
	if err != nil {
		resp.FailWithMsg(c, err.Error())
		return
	}
	req.ID = claims.ID

	user, err := repo.GetUserByID(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "验证失败")
		return
	}
	// 刷新令牌未过期，生成新的访问令牌
	token, err := jwt.GenerateTokenPair(req.ID, user.Email)
	if err != nil {
		resp.FailWithMsg(c, "验证失败")
		return
	}
	authResp := AuthResponse{
		AccessToken:  token.AccessToken,
		RefreshToken: token.RefreshToken,
		ExpiresIn:    token.ExpiresIn,
	}

	setRefreshTokenCookie(c, token.RefreshToken)

	// 刷新令牌未过期，返回新的访问令牌
	resp.OKWithData(c, authResp)
}

func FetchUserInfo(c *gin.Context) {
	id, err := getUserIDFromHeader(c)
	if err != nil {
		resp.FailWithMsg(c, err.Error())
		return
	}

	// 从数据库中查询用户
	user, err := repo.GetUserByID(uint(id))
	if err != nil {
		resp.FailWithMsg(c, "获取用户信息失败")
		return
	}
	// 构建用户信息响应
	userInfo := &UserInfo{
		ID:       user.ID,
		Email:    user.Email,
		Nickname: user.Nickname,
		Avatar:   user.Avatar,
		Identity: user.Identity,
	}
	// 返回用户信息
	resp.OKWithData(c, userInfo)
}

func Logout(c *gin.Context) {
	var req AuthRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "登出失败")
		return
	}

	// 从 Redis 中删除用户的 token 信息
	err = jwt.DeleteTokenFromRedis(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "登出失败")
		return
	}

	resp.OKWithMsg(c, "登出成功")
}

type RegisterRequest struct {
	Username        string `json:"username"`
	Email           string `json:"email"`
	Password        string `json:"password"`
	ConfirmPassword string `json:"confirm_password"`
	Nickname        string `json:"nickname"`
	Avatar          string `json:"avatar"`
	Identity        int    `json:"identity"`
	// Token           string `json:"token"`    // 登录凭证
}

func Register(c *gin.Context) {
	var req RegisterRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "注册失败")
		return
	}

	err = IsValidUsername(req.Username)
	if err != nil {
		resp.FailWithMsg(c, "注册失败，"+err.Error())
		return
	}
	// 检查用户名是否已存在
	_, err = repo.GetUserByUsername(req.Username)
	if err == nil {
		resp.FailWithMsg(c, "注册失败，用户名已被注册")
		return
	}

	// 检查邮箱是否已存在
	_, err = repo.GetUserByEmail(req.Email)
	if err == nil {
		resp.FailWithMsg(c, "注册失败，邮箱已被注册")
		return
	}

	// 检查密码是否符合要求
	err = IsValidPassword(req.Password)
	if err != nil {
		resp.FailWithMsg(c, "注册失败，"+err.Error())
		return
	}
	err = IsValidPassword(req.ConfirmPassword)
	if err != nil {
		resp.FailWithMsg(c, "注册失败，"+err.Error())
		return
	}

	// 检查密码是否一致
	if req.Password != req.ConfirmPassword {
		resp.FailWithMsg(c, "注册失败，两次密码不一致")
		return
	}

	pwd, err := HashPassword(req.Password)
	if err != nil {
		resp.FailWithMsg(c, "注册失败，密码加密失败")
		return
	}
	err = repo.CreateUser(&model.User{
		Username: req.Username,
		Email:    req.Email,
		Password: pwd,
		Nickname: req.Nickname,
		Avatar:   req.Avatar,
		Identity: req.Identity,
	})
	if err != nil {
		resp.FailWithMsg(c, "注册失败")
		return
	}
	resp.OKWithMsg(c, "注册成功")
}

type UpdateUserRequest struct {
	ID       uint   `json:"id"`
	Username string `json:"username"`
	Email    string `json:"email"`
	Nickname string `json:"nickname"`
	Identity int    `json:"identity"`
}

func UpdateUserInfo(c *gin.Context) {
	var req UpdateUserRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "更新用户信息失败")
		return
	}

	user, err := repo.GetUserByID(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "用户不存在，更新用户信息失败")
		return
	}

	if req.Username != "" {
		err = IsValidUsername(req.Username)
		if err != nil {
			resp.FailWithMsg(c, "更新用户信息失败，"+err.Error())
			return
		}
		user.Username = req.Username
	}
	if req.Email != "" {
		user.Email = req.Email
	}
	if req.Nickname != "" {
		user.Nickname = req.Nickname
	}
	user.Identity = req.Identity

	err = repo.UpdateUserInfo(user)
	// err = repo.UpdateUserInfo2(user)
	if err != nil {
		resp.FailWithMsg(c, "用户不存在，更新用户信息失败")
		return
	}
	resp.OKWithMsg(c, "更新用户信息成功")
}

type FetchUserListRequest struct {
	Keyword  string `json:"keyword"`
	Identity int    `json:"identity"`
	Page     int    `json:"page"`
	PageSize int    `json:"page_size"`
}

func FetchUserList(c *gin.Context) {
	var req FetchUserListRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "获取用户列表失败,参数错误")
		return
	}

	userList, total, err := repo.FetchUserList2(req.Keyword, req.Identity, req.Page, req.PageSize)
	if err != nil {
		resp.FailWithMsg(c, "获取用户列表失败")
		return
	}
	resp.OKWithData(c, gin.H{
		"user_list": userList,
		"total":     total,
	})
}

type DeleteUserRequest struct {
	ID       uint   `json:"id"`
	Username string `json:"username"`
}

func DeleteUser(c *gin.Context) {
	var req DeleteUserRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "删除用户失败")
		return
	}

	user, err := repo.GetUserByID(req.ID)
	if err != nil {
		if req.Username != "" {
			user, err = repo.GetUserByUsername(req.Username)
			if err != nil {
				resp.FailWithMsg(c, "用户不存在，删除用户失败")
				return
			}
		} else {
			resp.FailWithMsg(c, "用户不存在，删除用户失败")
			return
		}
	}

	err = repo.DeleteUserPermanently(user)
	if err != nil {
		resp.FailWithMsg(c, "删除用户失败")
		return
	}
	resp.OKWithMsg(c, "删除用户成功")
}

func GetCurrentUserProfile(c *gin.Context) {
	id, err := getUserIDFromHeader(c)
	if err != nil {
		resp.FailWithMsg(c, "获取用户信息失败")
		return
	}

	// 从数据库中查询用户
	user, err := repo.GetUserByID(uint(id))
	if err != nil {
		resp.FailWithMsg(c, "获取用户信息失败")
		return
	}

	// 构建用户信息响应
	userInfo := &UserInfo{
		ID:        user.ID,
		Email:     user.Email,
		Nickname:  user.Nickname,
		Username:  user.Username,
		Avatar:    user.Avatar,
		Identity:  user.Identity,
		CreatedAt: user.CreatedAt.Format(time.DateTime),
		UpdatedAt: user.UpdatedAt.Format(time.DateTime),
	}
	// 返回用户信息
	resp.OKWithData(c, userInfo)
}

type ChangePasswordRequest struct {
	ID              uint   `json:"id"`
	CurrentPassword string `json:"current_password"`
	NewPassword     string `json:"new_password"`
	ConfirmPassword string `json:"confirm_password"`
}

func ChangePassword(c *gin.Context) {
	var req ChangePasswordRequest
	id, err := getUserIDFromHeader(c)
	if err != nil {
		resp.FailWithMsg(c, err.Error())
		return
	}
	req.ID = id
	err = c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "修改密码失败,参数错误")
		return
	}

	// 检查当前密码是否正确
	user, err := repo.GetUserByID(req.ID)
	if err != nil {
		resp.FailWithMsg(c, "用户不存在，修改密码失败")
		return
	}
	if !CheckPassword(req.CurrentPassword, user.Password) {
		resp.FailWithMsg(c, "当前密码错误，修改密码失败")
		return
	}

	// 检查密码是否符合要求
	err = IsValidPassword(req.NewPassword)
	if err != nil {
		resp.FailWithMsg(c, "修改密码失败，"+err.Error())
		return
	}
	err = IsValidPassword(req.ConfirmPassword)
	if err != nil {
		resp.FailWithMsg(c, "修改密码失败，"+err.Error())
		return
	}

	// 检查新密码是否一致
	if req.NewPassword != req.ConfirmPassword {
		resp.FailWithMsg(c, "修改密码失败，两次密码不一致")
		return
	}

	// 加密新密码
	hashedPassword, err := HashPassword(req.NewPassword)
	if err != nil {
		resp.FailWithMsg(c, "修改密码失败")
		return
	}
	// 更新用户密码
	user.Password = hashedPassword
	err = repo.UpdateUserInfo(user)
	if err != nil {
		resp.FailWithMsg(c, "修改密码失败")
		return
	}
	resp.OKWithMsg(c, "修改密码成功")
}

type UpdateUserAvatarRequest struct {
	ID          uint
	ResetAvatar bool                  `form:"reset_avatar"`
	FileHeader  *multipart.FileHeader `form:"avatar"` // avatar
}

func UpdateUserAvatar(c *gin.Context) {
	var req UpdateUserAvatarRequest
	id, err := getUserIDFromHeader(c)
	if err != nil {
		resp.FailWithMsg(c, err.Error())
		return
	}
	req.ID = id

	err = c.ShouldBind(&req)
	if err != nil {
		resp.FailWithMsg(c, "更新用户头像失败,参数错误")
		return
	}

	user, err := repo.GetUserByID(req.ID) // 获得用户信息 检查用户是否存在
	if err != nil {
		resp.FailWithMsg(c, "更新用户头像失败,"+err.Error())
		return
	}

	/***************  重置/删除 用户头像、恢复默认头像  ***************/
	if req.ResetAvatar {
		tmp := user.Avatar
		user.Avatar = ""
		err = repo.UpdateUserInfo2(user, "avatar") // 调用时，明确指定要更新 avatar 字段
		if err != nil {
			resp.FailWithMsg(c, "重置用户头像失败,"+err.Error())
			return
		}
		RemoveFile(tmp)
		resp.OKWithMsg(c, "重置用户头像成功")
		return
	}

	/***************  修改用户头像  ***************/
	filePath, err := UploadFile(req.FileHeader, "./upload_data/avatar", &UploadOption{
		ID:       strconv.FormatUint(uint64(req.ID), 10),
		FileType: 1,
		MaxSize:  4 * 1024 * 1024, // 4MB
	})
	if err != nil {
		resp.FailWithMsg(c, "修改用户头像失败,"+err.Error())
		return
	}

	// 更新用户头像
	tmp := user.Avatar
	user.Avatar = filePath
	err = repo.UpdateUserInfo(user)
	if err != nil {
		resp.FailWithMsg(c, "修改用户头像失败,"+err.Error())
		return
	}
	RemoveFile(tmp)
	resp.OKWithData(c, user.Avatar) // 修改用户头像成功
}

// 忘记密码，重置密码流程
type ForgotPasswordRequest struct {
	Username         string `json:"username"`
	VerificationCode string `json:"verification_code"`
	NewPassword      string `json:"new_password"`
	ConfirmPassword  string `json:"confirm_password"`
}

// 忘记密码，发送验证码
func SendVerificationCode(c *gin.Context) {
	var req ForgotPasswordRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "发送验证码失败,参数错误")
		return
	}

	// 检查用户名是否存在
	user, err := repo.GetUserByUsername(req.Username)
	if err != nil {
		user, err = repo.GetUserByEmail(req.Username)
		if err != nil {
			resp.FailWithMsg(c, "发送验证码失败,用户不存在")
			return
		}
	}
	fmt.Println("忘记密码，检查用户名是否存在:", req.Username)

	// 发送验证码
	_, err = utils.Verifier.SendVerificationCode(user.Email, utils.BuildVerificationCode)
	if err != nil {
		resp.FailWithMsg(c, "发送验证码失败,"+err.Error())
		return
	}
	maskedEmail := MaskEmail(user.Email)
	resp.OKWithData(c, gin.H{"masked_email": maskedEmail})
}

// 忘记密码，重置密码
func ResetPassword(c *gin.Context) {
	var req ForgotPasswordRequest
	err := c.ShouldBindJSON(&req)
	if err != nil {
		resp.FailWithMsg(c, "重置密码失败,参数错误")
		return
	}

	user, err := repo.GetUserByUsername(req.Username)
	if err != nil {
		user, err = repo.GetUserByEmail(req.Username)
		if err != nil {
			resp.FailWithMsg(c, "用户不存在")
			return
		}
	}

	// 验证验证码
	err = utils.Verifier.VerifyCode(user.Email, req.VerificationCode)
	if err != nil {
		resp.FailWithMsg(c, "重置密码失败,验证码错误")
		return
	}

	// 检查新密码是否符合要求
	err = IsValidPassword(req.NewPassword)
	if err != nil {
		resp.FailWithMsg(c, "重置密码失败,新密码不符合要求")
		return
	}

	// 检查新密码和确认密码是否一致
	if req.NewPassword != req.ConfirmPassword {
		resp.FailWithMsg(c, "重置密码失败,两次密码不一致")
		return
	}

	// 哈希新密码
	hashedPassword, err := HashPassword(req.NewPassword)
	if err != nil {
		resp.FailWithMsg(c, "重置密码失败,密码哈希失败")
		return
	}

	// 更新用户密码
	user.Password = hashedPassword
	err = repo.UpdateUserInfo(user)
	if err != nil {
		resp.FailWithMsg(c, "重置密码失败,数据库更新失败")
		return
	}
	resp.OKWithMsg(c, "重置密码成功")
}
