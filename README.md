# Selene's Blog

This repository contains Selene's personal blog, built with Astro and adapted from
[Axi-Theme](https://github.com/Axi404/Axi-Theme).

The site content lives in:

- `src/content/blogs/` for blog posts
- `src/pages/about/index.md` for the About page
- `public/assets/img/posts/` for post images
- `public/output/pdf/` for downloadable PDF and TeX files

## Local development

```sh
pnpm install --frozen-lockfile
pnpm dev
```

## Production build

```sh
DEPLOYMENT_PLATFORM=github pnpm build
```

The production site is generated in `dist/` and deployed to GitHub Pages by the workflow
in `.github/workflows/deploy.yml`.
