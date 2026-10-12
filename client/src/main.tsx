import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import EchoesApp from "@/EchoesApp";
import "@/index.css";

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <EchoesApp />
  </StrictMode>
);
