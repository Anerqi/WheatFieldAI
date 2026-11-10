# 安全策略

## 报告问题

如发现本仓库存在凭据泄漏、依赖漏洞或其他安全问题，请通过 GitHub 仓库 **Security → Report a vulnerability** 私密提交，或联系仓库维护者（见 GitHub 个人主页）。请勿在公开 Issue 中披露可利用细节。

## 范围说明

- 本仓库代码树不含模型权重、数据集与真实图片；权重与数据集经 GitHub Release 分发，按各自来源条款管理（见 `THIRD_PARTY_NOTICES.md`）。
- 应用为本地 Streamlit 工作台，默认不对外网暴露；部署到公网前请自行加固鉴权、限流与上传限制。
