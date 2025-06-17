// @ts-check
/**
 * ESLint Flat Config (ESM)
 * 参考 typescript-eslint 官方示例，集成 Prettier
 */

import eslint from '@eslint/js';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import tsParser from '@typescript-eslint/parser';
import eslintConfigPrettier from 'eslint-config-prettier';
import globals from 'globals';

export default [
  // Ignore patterns（替代 .eslintignore）
  {
    ignores: ['**/node_modules/**', '**/dist/**'],
  },
  // 为所有文件提供 Node 环境内置全局变量，避免 no-undef
  {
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  },
  // JavaScript 基础推荐规则
  eslint.configs.recommended,
  // TypeScript 文件规则
  {
    files: ['**/*.ts'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        project: './tsconfig.json',
        tsconfigRootDir: import.meta.dirname,
      },
      sourceType: 'module',
      ecmaVersion: 'latest',
      globals: {
        ...globals.node,
      },
    },
    plugins: {
      '@typescript-eslint': tsPlugin,
    },
    rules: {
      ...tsPlugin.configs.recommended.rules,
      ...tsPlugin.configs.stylistic.rules,
    },
  },
  // 关闭与 Prettier 冲突的格式化规则
  eslintConfigPrettier,
];
