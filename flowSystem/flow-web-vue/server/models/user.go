package model

import (
	"fmt"
	"time"

	"gorm.io/gorm"
)

type User struct {
	// gorm.Model  // 没法直接自定义 json
	ID        uint           `gorm:"primarykey" json:"id"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	DeletedAt gorm.DeletedAt `gorm:"index" json:"deleted_at"` // GORM框架的标准软删除字段（逻辑删除），软删除时会设置该字段

	Username string `gorm:"unique" json:"username"`
	Email    string `gorm:"unique" json:"email"`
	Password string `json:"password"`
	Nickname string `json:"nickname"`
	Avatar   string `json:"avatar"`
	Identity int    `json:"identity"` // 身份，如 1 管理员等、2 普通用户
	Token    string `json:"token"`    // 登录凭证
}

func (u *User) TableName() string {
	return "user"
}

func (u *User) Println() {
	fmt.Println(">>>>>>>>>> User:")
	fmt.Println(">>>>>>>>>>>>>> ID:", u.ID)
	fmt.Println(">>>>>>>>>>>>>> CreatedAt:", u.CreatedAt)
	fmt.Println(">>>>>>>>>>>>>> UpdatedAt:", u.UpdatedAt)
	fmt.Println(">>>>>>>>>>>>>> DeletedAt:", u.DeletedAt)
	fmt.Println(">>>>>>>>>>>>>> Email:", u.Email)
	fmt.Println(">>>>>>>>>>>>>> Password:", u.Password)
	fmt.Println(">>>>>>>>>>>>>> Nickname:", u.Nickname)
	fmt.Println(">>>>>>>>>>>>>> Avatar:", u.Avatar)
	fmt.Println(">>>>>>>>>>>>>> Identity:", u.Identity)
	fmt.Println(">>>>>>>>>>>>>> Token:", u.Token)
}
