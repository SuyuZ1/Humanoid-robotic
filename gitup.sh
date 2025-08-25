cat >> ~/.bashrc <<'EOF'

gitup () {
  msg="$1"
  [ -z "$msg" ] && msg="update"
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
  git add .
  git commit -m "$msg" || true
  git push -u origin "$branch"
}

EOF


# source ~/.bashrc






# # === push existing local project via SSH ===
# PROJECT_DIR="/cpfs01/zsy_workspace/humanoid/Humanoid-robotic"  # 改成你的本地项目路径
# OWNER="SuyuZ1"                                                # GitHub 用户名
# REPO="Humanoid-robotic"                                       # 目标仓库名（已存在）
# BRANCH="main"

# cd "$PROJECT_DIR"

# # 如无 git 初始化则 init
# [ -d .git ] || git init

# # 确保有一次提交
# [ -f README.md ] || echo "# $REPO" > README.md
# git add .
# git commit -m "init or update" || true    # 若无变更会提示 nothing to commit 可忽略

# # 绑定 SSH 远程并推送
# git remote remove origin 2>/dev/null || true
# git remote add origin git@github.com:${OWNER}/${REPO}.git
# git branch -M "$BRANCH"
# git push -u origin "$BRANCH"
