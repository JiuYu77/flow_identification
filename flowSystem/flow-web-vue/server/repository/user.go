package repo

import (
	model "flow-server/models"

	"gorm.io/gorm"
)

func GetUserByEmail(email string) (*model.User, error) {
	var user model.User
	err := DB.Where("email = ?", email).First(&user).Error
	if err != nil {
		return nil, err
	}
	return &user, nil
}
func GetUserByID(id uint) (*model.User, error) {
	var user model.User
	err := DB.Where("id = ?", id).First(&user).Error
	if err != nil {
		return nil, err
	}
	return &user, nil
}

func CreateUser(user *model.User) error {
	return DB.Create(user).Error
}

func GetUserByUsername(username string) (*model.User, error) {
	var user model.User
	err := DB.Where("username = ?", username).First(&user).Error
	if err != nil {
		return nil, err
	}
	return &user, nil
}

// 会更新表中所有行，除非 user 结构体中包含 gorm:"primaryKey" 标签的主键字段（如 ID），
// 且 user.ID != 0（此时 GORM 才会自动补全 WHERE id=?）。
//
// 会忽略 传入的零值字段（如空字符串、0、false），仅更新非零值字段。
func UpdateUserInfo(user *model.User) error {
	return DB.Updates(user).Error
}

// 支持指定更新字段
func UpdateUserInfo2(user *model.User, updateFields ...string) error {
	tx := DB.Model(user)
	if len(updateFields) > 0 {
		tx = tx.Select(updateFields) // 显式指定要更新的字段
	}
	return tx.Updates(user).Error
}

func FetchUserList(keyword string, identity int) ([]*model.User, error) {
	var userList []*model.User
	var tx *gorm.DB

	if keyword == "" && identity == -1 {
		tx = DB.Find(&userList)
	} else if keyword == "" && identity != -1 {
		tx = DB.Where("identity = ?", identity).Find(&userList)
	} else if keyword != "" && identity == -1 {
		tx = DB.Where("username = ? or email = ? or nickname = ?", keyword, keyword, keyword).Find(&userList)
	} else {
		tx = DB.Where(
			"(username = ? or email = ? or nickname = ?) and identity = ?",
			keyword, keyword, keyword, identity,
		).Find(&userList)
	}

	if tx.Error != nil {
		return nil, tx.Error
	}

	return userList, nil
}

// FetchUserList2 分页查询用户列表
//
// # Args
//   - keyword: 搜索关键词（用户名、邮箱、昵称）
//   - identity: 用户身份（-1 表示查询所有用户）
//   - page: 当前页码
//   - pageSize: 每页数量
func FetchUserList2(keyword string, identity int, page, pageSize int) ([]*model.User, int64, error) {
	var userList []*model.User
	var tx *gorm.DB
	var total int64

	// 模糊查询：使用 LIKE 操作符，%keyword% 格式实现前后模糊匹配
	keywordPattern := "%" + keyword + "%"

	// 构建查询条件
	if keyword == "" && identity == -1 {
		tx = DB
	} else if keyword == "" && identity != -1 {
		tx = DB.Where("identity = ?", identity)
	} else if keyword != "" && identity == -1 {
		tx = DB.Where("username LIKE ? OR email LIKE ? OR nickname LIKE ?",
			keywordPattern, keywordPattern, keywordPattern)
	} else {
		tx = DB.Where(
			"(username LIKE ? OR email LIKE ? OR nickname LIKE ?) AND identity = ?",
			keywordPattern, keywordPattern, keywordPattern, identity)
	}

	if tx.Error != nil {
		return nil, 0, tx.Error
	}

	// 获取总数
	if err := tx.Model(&model.User{}).Count(&total).Error; err != nil {
		return nil, 0, err
	}

	// 分页查询
	tx = tx.Offset((page - 1) * pageSize).Limit(pageSize).Find(&userList)
	if tx.Error != nil {
		return nil, 0, tx.Error
	}

	return userList, total, nil
}

// DeleteUser 删除用户，软删除（逻辑删除）
func DeleteUser(user *model.User) error {
	return DB.Delete(user).Error
}

// DeleteUserPermanently 物理删除用户（永久删除）
func DeleteUserPermanently(user *model.User) error {
	return DB.Unscoped().Delete(user).Error
}

// DeleteUserByIDPermanently 根据ID物理删除用户
func DeleteUserByIDPermanently(id uint) error {
	return DB.Unscoped().Delete(&model.User{}, id).Error
}

// DeleteUserByUsernamePermanently 根据用户名物理删除用户
func DeleteUserByUsernamePermanently(username string) error {
	return DB.Unscoped().Where("username = ?", username).Delete(&model.User{}).Error
}
