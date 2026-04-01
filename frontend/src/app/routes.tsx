import { createBrowserRouter } from "react-router";
import { IndexPage } from "./pages/index-page";
import { ReportPage } from "./pages/report-page";
import { SettingsPage } from "./pages/settings-page";
import { AboutPage } from "./pages/about-page";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: IndexPage,
  },
  {
    path: "/report",
    Component: ReportPage,
  },
  {
    path: "/settings",
    Component: SettingsPage,
  },
  {
    path: "/about",
    Component: AboutPage,
  },
]);